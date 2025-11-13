import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from surrogates.surrogate_eval import prepare_data
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent
from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive
import pyro.distributions as dist
from torch.utils.data import ConcatDataset
import toml
from transformers import AutoTokenizer, AutoModel
import subprocess
from codec import Codec
from eval import get_optimizer, get_scheduler
import traceback
from torch.optim import lr_scheduler
from transformers import AutoTokenizer, AutoModel
import json
from surrogates.surrogate_dataset import build_dataset
import os
import hashlib
import time
import re

USER = os.getenv("USER", "psomu3")
cwd = os.path.dirname(os.getcwd())
cfg = toml.load(os.path.join(cwd, "conf.toml"))

genome_encoding_strat = cfg["codec"]['genome_encoding_strat']
num_classes = cfg["model"]['num_classes']
num_loss_comp = cfg["model"]['num_loss_components']

# Pre-analyzer job configuration (similar to pipeline.py)
PREANALYZER_JOB_NAME = 'preanalyzer_codestral'
PREANALYZER_NODES = 1
PREANALYZER_CORES = 8
PREANALYZER_MEM = '32GB'
PREANALYZER_TIME = '02:00:00'
PREANALYZER_ENV = 'nas'
PREANALYZER_GPUS = ["V100-16GB", "V100-32GB", "L40S", "A100-40GB", "H100", "H200"]

def create_preanalyzer_job_file(dataset_file, dataset_name, status_dir, start_idx=0):
    """Create an SBATCH job file for running the pre-analyzer"""
    job_file = f"{PREANALYZER_JOB_NAME}_{dataset_name}.job"
    
    batch_script = f"""#!/bin/bash
#SBATCH --job-name={PREANALYZER_JOB_NAME}_{dataset_name}
#SBATCH --nodes={PREANALYZER_NODES}
#SBATCH -G 1
#SBATCH --cpus-per-task={PREANALYZER_CORES}
#SBATCH --mem={PREANALYZER_MEM}
#SBATCH --time={PREANALYZER_TIME}
#SBATCH --output={status_dir}/{dataset_name}_preanalyzer.%j.log
#SBATCH --error={status_dir}/{dataset_name}_preanalyzer_error.%j.log
#SBATCH --constraint="{'|'.join(PREANALYZER_GPUS)}"

module load anaconda3/2023.03
module load cuda/12.1.1

# Execute the pre-analyzer script
conda run -n {PREANALYZER_ENV} --no-capture-output python -u -m surrogates.codestral_preanalyzer --dataset_file {dataset_file} --dataset_name {dataset_name} --status_dir {status_dir} --start_idx {start_idx}
"""
    
    with open(job_file, 'w') as f:
        f.write(batch_script)
    
    return job_file

def load_preanalyzer_status(status_file):
    """Load the current status of a pre-analyzer"""
    if not os.path.exists(status_file):
        return {'current_idx': 0, 'status': 'not_started'}
    
    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except:
        return {'current_idx': 0, 'status': 'error'}

def load_failure_list(failure_file):
    """Load the list of failed genome hashes"""
    failed_hashes = set()
    if os.path.exists(failure_file):
        try:
            with open(failure_file, 'r') as f:
                for line in f:
                    if line.strip():
                        failure_data = json.loads(line.strip())
                        failed_hashes.add(failure_data['genome_hash'])
        except Exception as e:
            print(f"Error loading failure file {failure_file}: {e}")
    
    return failed_hashes

def get_genome_hash(genome_str):
    """Generate a simple hash for the genome string"""
    return hashlib.md5(str(genome_str).encode()).hexdigest()[:10]

def submit_preanalyzer_job(dataset_file, dataset_name, status_dir, start_idx=0):
    """Submit a pre-analyzer job and return the job ID"""
    job_file = create_preanalyzer_job_file(dataset_file, dataset_name, status_dir, start_idx)
    
    print(f"Submitting pre-analyzer job for {dataset_name}...")
    sbatch_result = os.popen(f"sbatch {job_file}").read()
    
    # Parse job ID from sbatch output
    match = re.search(r'Submitted batch job (\d+)', sbatch_result)
    if match:
        job_id = match.group(1)
        print(f"  Job ID: {job_id}")
        return job_id
    else:
        print(f"  Failed to submit job: {sbatch_result}")
        return None

def check_job_status(job_id):
    """Check if a SLURM job is still running"""
    if not job_id:
        return False
    
    try:
        squeue_result = subprocess.run(['squeue', '-j', job_id], 
                                     capture_output=True, text=True, timeout=30)
        # If job is in the queue, squeue will return it
        return job_id in squeue_result.stdout
    except:
        return False

def monitor_preanalyzer(dataset_name, status_dir, dataset_file, max_restarts=3):
    """Monitor a pre-analyzer process and restart it if it fails"""
    status_file = os.path.join(status_dir, f"{dataset_name}_status.json")
    restart_count = 0
    current_job_id = None
    
    while restart_count <= max_restarts:
        # Check current status
        status = load_preanalyzer_status(status_file)
        
        if status['status'] == 'completed':
            print(f"Pre-analyzer for {dataset_name} completed successfully")
            return True
        
        # If not running and not completed, (re)start the job
        if not check_job_status(current_job_id):
            start_idx = status.get('current_idx', 0)
            
            if status['status'] == 'fatal_error':
                print(f"Pre-analyzer for {dataset_name} had fatal error, restarting from index {start_idx}")
                restart_count += 1
                if restart_count > max_restarts:
                    print(f"Max restarts ({max_restarts}) reached for {dataset_name}")
                    return False
            
            current_job_id = submit_preanalyzer_job(dataset_file, dataset_name, status_dir, start_idx)
            if not current_job_id:
                print(f"Failed to submit job for {dataset_name}")
                return False
        
        # Wait before checking again
        time.sleep(30)
    
    return False

def monitor_preanalyzers(datasets, status_dir, max_restarts=3):
    """Monitor multiple pre-analyzer processes simultaneously"""
    
    print(f"\n{'='*60}", flush=True)
    print("STARTING PRE-ANALYZER MONITORING", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Initialize tracking for each dataset
    dataset_info = {}
    for dataset_name, dataset_file in datasets.items():
        dataset_info[dataset_name] = {
            'dataset_file': dataset_file,
            'status_file': os.path.join(status_dir, f"{dataset_name}_status.json"),
            'restart_count': 0,
            'current_job_id': None,
            'completed': False,
            'failed': False
        }
    
    print(f"Will monitor {len(datasets)} pre-analyzers", flush=True)
    print(f"Status directory: {status_dir}", flush=True)
    
    # Check if pre-analyzers already completed
    all_already_done = True
    for dataset_name, info in dataset_info.items():
        status = load_preanalyzer_status(info['status_file'])
        print(f"  {dataset_name}: current status = {status['status']}", flush=True)
        if status['status'] == 'completed':
            info['completed'] = True
            print(f"    ✓ Already completed", flush=True)
        else:
            all_already_done = False
    
    if all_already_done:
        print("\n✓ All pre-analyzers already completed! Skipping job submission.", flush=True)
        return True
    
    # Submit initial jobs for datasets that aren't completed
    print("\nSubmitting jobs for incomplete datasets...", flush=True)
    for dataset_name, info in dataset_info.items():
        if info['completed']:
            continue
            
        print(f"  Submitting job for {dataset_name}...", flush=True)
        info['current_job_id'] = submit_preanalyzer_job(
            info['dataset_file'], dataset_name, status_dir, 0
        )
        if not info['current_job_id']:
            print(f"    ✗ Failed to submit job", flush=True)
            info['failed'] = True
        else:
            print(f"    ✓ Submitted with job ID: {info['current_job_id']}", flush=True)
    
    print("\nEntering monitoring loop...", flush=True)
    iteration = 0
    
    # Monitor all jobs
    while True:
        iteration += 1
        print(f"\n--- Monitoring iteration {iteration} ---", flush=True)
        
        all_completed = True
        any_active = False
        
        for dataset_name, info in dataset_info.items():
            if info['completed'] or info['failed']:
                continue
                
            # Check current status from file
            status = load_preanalyzer_status(info['status_file'])
            current_idx = status.get('current_idx', 0)
            
            print(f"{dataset_name}: status={status['status']}, idx={current_idx}, job_id={info['current_job_id']}", flush=True)
            
            if status['status'] == 'completed':
                print(f"  ✓ Pre-analyzer for {dataset_name} completed successfully", flush=True)
                info['completed'] = True
                continue
                
            all_completed = False
            
            # Check if job is still running
            job_running = check_job_status(info['current_job_id'])
            print(f"  Job running: {job_running}", flush=True)
            
            if not job_running:
                # Job stopped - check if it completed or failed
                if status['status'] == 'fatal_error':
                    print(f"  ⚠ Pre-analyzer had fatal error, restart count: {info['restart_count']}", flush=True)
                    info['restart_count'] += 1
                    
                    if info['restart_count'] > max_restarts:
                        print(f"  ✗ Max restarts ({max_restarts}) reached", flush=True)
                        info['failed'] = True
                        continue
                    
                    # Restart the job
                    start_idx = status.get('current_idx', 0)
                    print(f"  Restarting from index {start_idx}...", flush=True)
                    info['current_job_id'] = submit_preanalyzer_job(
                        info['dataset_file'], dataset_name, status_dir, start_idx
                    )
                    
                    if not info['current_job_id']:
                        print(f"  ✗ Failed to restart job", flush=True)
                        info['failed'] = True
                        continue
                    else:
                        print(f"  ✓ Restarted with job ID: {info['current_job_id']}", flush=True)
                        
                elif status['status'] == 'completed':
                    # Status updated to completed
                    print(f"  ✓ Status file shows completion", flush=True)
                    info['completed'] = True
                    continue
                else:
                    # Job may have just finished, give it time to update status
                    print(f"  ? Job ended with status: {status['status']}, waiting for status update...", flush=True)
                    
            if not info['failed']:
                any_active = True
        
        # Check if all are done
        if all_completed:
            print("\n" + "="*60, flush=True)
            print("✓ ALL PRE-ANALYZERS COMPLETED SUCCESSFULLY!", flush=True)
            print("="*60 + "\n", flush=True)
            break
            
        # Check if all failed
        if not any_active:
            failed_datasets = [name for name, info in dataset_info.items() if info['failed']]
            if failed_datasets:
                print(f"\n✗ Some pre-analyzers failed: {failed_datasets}", flush=True)
                return False
        
        # Print status summary
        completed_count = sum(1 for info in dataset_info.values() if info['completed'])
        failed_count = sum(1 for info in dataset_info.values() if info['failed'])
        running_count = len(dataset_info) - completed_count - failed_count
        
        print(f"\n📊 Summary: {completed_count} completed, {running_count} running, {failed_count} failed", flush=True)
        
        # Wait before next check
        print(f"Waiting 60 seconds before next check...", flush=True)
        time.sleep(60)  # Check every minute
    
    # Return success if all completed
    return all([info['completed'] for info in dataset_info.values()])

# Initialize Codestral model for embedding extraction
def initialize_codestral_model():
    """Initialize the Codestral model for generating code embeddings"""
    # Using Codestral-22B-v0.1 or a similar model
    # Note: You may need to adjust the model name based on availability
    model_name = "mistralai/Codestral-22B-v0.1"  # or appropriate HuggingFace model
    
    print(f"Loading Codestral model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Clear CUDA cache after model initialization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return tokenizer, model, device


def create_model_text_representation(model, model_dict):
    """
    Create a structured text representation of the PyTorch model
    matching exactly what's printed in decode_and_inspect_model
    """
    
    # Start building the text representation
    text_parts = []
    
    # 3. Model architecture
    text_parts.append("Thoroughly analyze the object detection model architecture:")
    text_parts.append(str(model))
    
    # 4. Optimizer configuration (matching the exact format from decode_and_inspect_model)
    try:
        params = model.parameters()
        optimizer = get_optimizer(params, model_dict)
        text_parts.append(f"{type(optimizer).__name__}(")
        optimizer_params = optimizer.param_groups[0]
        for key, value in optimizer_params.items():
            if key != 'params':  # 'params' is a list of the model's tensors
                text_parts.append(f"  {key}: {value}")
        text_parts.append(")")
    except Exception as opt_error:
        text_parts.append(f"Could not create optimizer: {opt_error}")

    # 5. Scheduler configuration (matching the exact format from decode_and_inspect_model)
    try:
        scheduler = get_scheduler(optimizer, model_dict, num_epochs=10, batch_size=32)
        text_parts.append(f"{type(scheduler).__name__}(")
        scheduler_params = scheduler.state_dict()
        for key, value in scheduler_params.items():
            text_parts.append(f"  {key}: {value}")
        text_parts.append(")")
    except Exception as sched_error:
        text_parts.append(f"Could not create scheduler: {sched_error}")

    # 6. Loss component weights (matching the exact format from decode_and_inspect_model)
    extra_vector = np.array(list(map(float, model_dict.get('loss_weights', 'N/A'))))
    
    # Join all parts with newlines
    full_text = "\n".join(text_parts)
    
    return full_text, extra_vector

def get_codestral_embedding(text, tokenizer, model, device, max_length=8192):
    """
    Generate embeddings using Codestral model, following the hidden state approach from the paper
    """
    try:
        # Tokenize the input text
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True, 
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Clear CUDA cache before processing to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Extract hidden states (following the paper's approach)
            # Use the last token's hidden state from the final layer
            hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
            
            # Get the last token's representation (similar to the paper's approach)
            last_token_embedding = hidden_states[0, -1, :].cpu()  # Move to CPU immediately
            
            # Also try mean pooling as alternative (mentioned in the paper)
            mean_pooled_embedding = torch.mean(hidden_states[0], dim=0).cpu()  # Move to CPU immediately
            
            # Clear CUDA cache after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                'last_token': last_token_embedding.numpy(),
                'mean_pooled': mean_pooled_embedding.numpy(),
                'hidden_states': hidden_states.cpu().numpy()
            }
    
    except torch.cuda.OutOfMemoryError as cuda_e:
        print(f"CUDA out of memory error: {cuda_e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"Error generating Codestral embedding: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None




def decode_and_inspect_model(encoded_genome, codec, num_loss_components=4):
    """
    Takes an encoded genome vector, decodes it to a PyTorch model, and prints its structure
    """
    print("="*80, flush=True)
    print("DECODING GENOME TO PYTORCH MODEL", flush=True)
    print("="*80, flush=True)
    
    try:
        # Decode the genome using the codec
        model_dict = codec.decode_genome(encoded_genome, num_loss_components)
        model = model_dict['model']
        
        print(f"Model Type: {type(model)}", flush=True)
        print(f"Model Dict Keys: {model_dict.keys()}", flush=True)
        # print("\nMODEL ARCHITECTURE:", flush=True)
        # print("-" * 40, flush=True)
        # print(model, flush=True)
        
        # try:
        #     # Get model parameters for optimizer creation
        #     params = model.parameters()
        #     optimizer = get_optimizer(params, model_dict)
        #     print(f"{type(optimizer).__name__}(", flush=True)
        #     optimizer_params = optimizer.param_groups[0]
        #     for key, value in optimizer_params.items():
        #         if key != 'params': # 'params' is a list of the model's tensors
        #             print(f"  {key}: {value}")
        #     print(")", flush=True)
        # except Exception as opt_error:
        #     print(f"Could not create optimizer: {opt_error}")

        # # Print scheduler configuration
        # try:
        #     scheduler = get_scheduler(optimizer, model_dict, num_epochs=10, batch_size=32)
        #     print(f"{type(scheduler).__name__}(", flush=True)
        #     scheduler_params = scheduler.state_dict()
        #     for key, value in scheduler_params.items():
        #         print(f"  {key}: {value}")
        #     print(")", flush=True)
        # except Exception as sched_error:
        #     print(f"Could not create scheduler: {sched_error}")

        # print(f"LossComponentWeights: {tuple(map(float, model_dict.get('loss_weights', 'N/A')))}", flush=True)

        # print("\nFORWARD TRACE", flush=True)
        # print("-" * 40, flush=True)
        # model.eval()
        
        # # Try different tracing methods in order of preference
        # traced = False
        
        # # Method 1: Try torch.fx symbolic trace (works for simpler models)
        # try:
        #     traced_model = torch.fx.symbolic_trace(model)
        #     print("PyTorch FX Symbolic Trace:")
        #     print(traced_model.code, flush=True)
        #     traced = True
        # except Exception as fx_error:
        #     print(f"FX symbolic trace failed: {fx_error}")
        
        # # Method 2: Use TorchScript tracing as fallback
        # if not traced:
        #     try:
        #         print("\nTrying TorchScript tracing...")
        #         dummy_input = torch.randn(1, 3, 224, 224)
        #         if torch.cuda.is_available():
        #             dummy_input = dummy_input.cuda()
        #             model = model.cuda()
                
        #         with torch.no_grad():
        #             traced_model = torch.jit.trace(model, dummy_input)
        #             print("TorchScript Trace Graph:")
        #             print(traced_model.graph, flush=True)
        #         traced = True
        #     except Exception as jit_error:
        #         print(f"TorchScript tracing failed: {jit_error}")
        #         traceback.print_exc()
        
        # # Method 3: Manual forward hook inspection as final fallback
        # if not traced:
        #     print("\nUsing forward hooks to inspect model execution...")
            
        #     def hook_fn(module, input, output):
        #         module_name = module.__class__.__name__
        #         input_shape = input[0].shape if isinstance(input, (tuple, list)) and len(input) > 0 else "Unknown"
        #         if isinstance(output, torch.Tensor):
        #             output_shape = output.shape
        #         elif isinstance(output, (tuple, list)):
        #             output_shape = [o.shape if isinstance(o, torch.Tensor) else type(o) for o in output]
        #         elif isinstance(output, dict):
        #             output_shape = {k: v.shape if isinstance(v, torch.Tensor) else type(v) for k, v in output.items()}
        #         else:
        #             output_shape = type(output)
        #         print(f"  {module_name}: {input_shape} -> {output_shape}")
            
        #     # Register hooks for all modules
        #     hooks = []
        #     for name, module in model.named_modules():
        #         if name:  # Skip root module
        #             hook = module.register_forward_hook(hook_fn)
        #             hooks.append(hook)
            
        #     try:
        #         dummy_input = torch.randn(1, 3, 224, 224)
        #         if torch.cuda.is_available():
        #             dummy_input = dummy_input.cuda()
        #             model = model.cuda()
                
        #         print("Forward pass execution trace:")
        #         with torch.no_grad():
        #             output = model(dummy_input)
        #         print(f"Final output type: {type(output)}")
        #         if isinstance(output, dict):
        #             for k, v in output.items():
        #                 if isinstance(v, torch.Tensor):
        #                     print(f"  {k}: {v.shape}")
        #                 else:
        #                     print(f"  {k}: {type(v)}")
        #     except Exception as hook_error:
        #         print(f"Forward hook inspection failed: {hook_error}")
        #     finally:
        #         # Clean up hooks
        #         for hook in hooks:
        #             hook.remove()
        
        # print("\n" + "="*60)
        
        # print(f"\nMODEL PARAMETERS:", flush=True)
        # print("-" * 40, flush=True)
        # total_params = 0
        # for name, param in model.named_parameters():
        #     param_count = param.numel()
        #     total_params += param_count
        #     print(f"{name}: {param.shape} ({param_count:,} parameters)", flush=True)
        
        # print(f"\nTotal Parameters: {total_params:,}", flush=True)
        
        # # Print model structure in a more detailed way
        # print(f"\nDETAILED MODEL STRUCTURE:", flush=True)
        # print("-" * 40, flush=True)
        # for name, module in model.named_modules():
        #     if name:  # Skip the root module
        #         print(f"{name}: {module}", flush=True)
        
        # If available, print loss weights and other config
        # Generate Codestral embedding
        print("\nGENERATING CODESTRAL EMBEDDING:", flush=True)
        print("-" * 40, flush=True)
        
        # Initialize Codestral model
        tokenizer, codestral_model, codestral_device = initialize_codestral_model()
        
        if tokenizer is not None and codestral_model is not None:
            # Create text representation of the model
            model_text, extra_vector = create_model_text_representation(model, model_dict)
            
            print(f"Model text representation length: {len(model_text)} characters", flush=True)
            print(f"First 500 characters of text representation:", flush=True)
            print(model_text[:500] + "..." if len(model_text) > 500 else model_text, flush=True)
            
            # Generate embedding using Codestral
            codestral_embeddings = get_codestral_embedding(model_text, tokenizer, codestral_model, codestral_device)
            
            if codestral_embeddings is not None:
                print(f"\nCodestral embedding dimensions:", flush=True)
                print(f"  Last token embedding: {codestral_embeddings['last_token'].shape}", flush=True)
                print(f"  Mean pooled embedding: {codestral_embeddings['mean_pooled'].shape}", flush=True)
                print(f"  Full hidden states: {codestral_embeddings['hidden_states'].shape}", flush=True)

                codestral_embeddings['last_token'] = np.concatenate([codestral_embeddings['last_token'], extra_vector], axis=0)
                codestral_embeddings['mean_pooled'] = np.concatenate([codestral_embeddings['mean_pooled'], extra_vector], axis=0)
                print(f"  Updated Last token embedding: {codestral_embeddings['last_token'].shape}", flush=True)
                print(f"  Updated Mean pooled embedding: {codestral_embeddings['mean_pooled'].shape}", flush=True)
                
                # Save the embeddings for future use
                embedding_save_path = f"/storage/ice-shared/vip-vvk/data/AOT/{USER}/codestral/codestral_embedding_{abs(hash(encoded_genome))}.npz"
                np.savez(
                    embedding_save_path,
                    last_token=codestral_embeddings['last_token'],
                    mean_pooled=codestral_embeddings['mean_pooled'],
                    model_text=model_text,
                    genome=encoded_genome
                )
                print(f"Saved Codestral embeddings to: {embedding_save_path}", flush=True)
                
                # Following the paper's approach, we'll use the last token embedding as primary
                primary_embedding = codestral_embeddings['last_token']
                print(f"\nPrimary embedding (last token) statistics:", flush=True)
                print(f"  Shape: {primary_embedding.shape}", flush=True)
                print(f"  Mean: {np.mean(primary_embedding):.4f}", flush=True)
                print(f"  Std: {np.std(primary_embedding):.4f}", flush=True)
                print(f"  Min: {np.min(primary_embedding):.4f}", flush=True)
                print(f"  Max: {np.max(primary_embedding):.4f}", flush=True)
                
                return model, model_dict, codestral_embeddings
            else:
                print("Failed to generate Codestral embeddings", flush=True)
                return model, model_dict, None
        else:
            print("Failed to initialize Codestral model", flush=True)
            return model, model_dict, None
        
    except Exception as e:
        print(f"Error decoding genome: {e}", flush=True)
        # print entire error
        traceback.print_exc()
        print(f"Genome type: {type(encoded_genome)}", flush=True)
        if hasattr(encoded_genome, 'shape'):
            print(f"Genome shape: {encoded_genome.shape}", flush=True)
        return None, None, None

# Initialize codec
print("Initializing Codec...", flush=True)
codec = Codec(num_classes, genome_encoding_strat=genome_encoding_strat)
print(f"Codec initialized with {num_classes} classes and {genome_encoding_strat} encoding strategy", flush=True)

# Get the first genome from training data
# sample_genome = """RetinaNet_Head(ReLU_2D(ConvNeXt(Dropout_2D(LeakyReLU_2D(IN0, 45.404551619753086), toProbFloat(64.12191929065479)), dummyOp(dummyOp(0)), dummyOp(1))), SGD(toPNorm(protectedDiv(toBoundedFloat(52.76595922670645), toProbFloat(0.6763818489615296))), toBoundedFloat(protectedSub(mul(99.46373058704523, 1.339714120118456), add(2.5406144376662736, 0.044832281908226856))), toBoundedFloat(mul(add(46.12303964781302, 0.48371801392204916), protectedDiv(70.47990104688951, 16.607548644437607))), toProbFloat(toProbFloat(protectedSub(1.923462795166177, 1.2171335332978759)))), MultiStepLR(add(toBoundedFloat(4.641602310869308), toBoundedFloat(mul(0.5624384936038089, 0.39472543140164207)))), toProbFloat(toProbFloat(toBoundedFloat(add(2.272574908891945, 1.716923393926169)))), toProbFloat(protectedSub(protectedSub(protectedDiv(2.8037916425084486, 25.76110674879022), toProbFloat(30.9956056758934)), toPNorm(37.722342358957114))), toProbFloat(toBoundedFloat(mul(mul(48.04336970233029, 36.91190840967564), protectedDiv(0.1629064550293845, 11.110526435819057)))), toProbFloat(toPNorm(2.658560081309097)), toProbFloat(mul(toProbFloat(toProbFloat(66.43472067449122)), mul(add(40.44379753554487, 82.09753735711185), toBoundedFloat(2.3394223508253784)))), toProbFloat(toBoundedFloat(toProbFloat(toPNorm(52.07275973157193)))), toProbFloat(toProbFloat(toProbFloat(add(1.8178114756218984, 89.92179725693586)))))"""
sample_genome = "RetinaNet_Head(Sigmoid_2D(Upsample_2D(Upsample_2D(LazyConvTranspose2d(Threshold_2D(IN0, 52.04076851212076, 25.525353970015942), toChannel(64), toKernel(97), toKernel(76), toStride(36), toStride(24), toPadding(60), toPadding(36), dummyOp(2), toDilation(29), toDilation(26), toGroup(16)), toBoundedFloat(toProbFloat(0.23419980808057805)), dummyOp(dummyOp(1))), toBoundedFloat(toBoundedFloat(toPNorm(87.554447546241))), dummyOp(dummyOp(dummyOp(1))))), Adagrad(add(mul(mul(protectedSub(79.34925885938583, 87.59522198292993), toProbFloat(56.6889274090664)), toPNorm(protectedDiv(46.420249655585366, 71.81085867235583))), protectedSub(protectedDiv(protectedDiv(13.884316745217017, 33.64332758940169), protectedDiv(2.3066889770646055, 38.23551562746258)), add(protectedDiv(53.346037423620665, 41.598967693027845), toBoundedFloat(33.693542430837866)))), mul(protectedSub(add(toBoundedFloat(2.181121006354955), toPNorm(31.104979280976053)), mul(toProbFloat(67.26879267533447), protectedSub(0.22786088010573624, 85.25768807924183))), protectedDiv(add(toBoundedFloat(1.3443297650246122), toProbFloat(33.68836649024036)), protectedSub(mul(0.33330945505966114, 2.5899942650257235), mul(44.002293924526406, 1.133805227107364))))), CosineAnnealingWarmRestarts(toDilation(protectedDiv(add(add(26, 17), protectedDiv(97, 74)), mul(protectedSub(54, 56), protectedDiv(85, 76)))), protectedDiv(mul(protectedSub(protectedDiv(25, 34), protectedSub(84, 40)), mul(protectedSub(3, 15), protectedSub(21, 76))), protectedSub(add(add(59, 12), protectedSub(16, 9)), add(add(11, 80), protectedSub(64, 2)))), add(add(mul(toProbFloat(42.31626772838582), add(26.9990308454939, 1.903413946914873)), toBoundedFloat(toPNorm(1.3262364101670152))), toProbFloat(toPNorm(protectedSub(51.29626311575222, 19.761159295608554))))), toProbFloat(toPNorm(toProbFloat(add(protectedDiv(7.665232612410422, 54.42871982127016), toProbFloat(20.105834281003876))))), toProbFloat(protectedSub(toPNorm(protectedDiv(protectedSub(0.37869423261011714, 0.3873229201126863), toProbFloat(38.1539775429259))), add(protectedDiv(protectedDiv(71.1172692373085, 1.3707920637635616), protectedDiv(0.7922488614576606, 2.742566923426746)), toBoundedFloat(toBoundedFloat(83.41137426324345))))), toProbFloat(toBoundedFloat(toBoundedFloat(mul(toBoundedFloat(87.54047401467648), toProbFloat(1.0242330623955083))))), toProbFloat(toProbFloat(toProbFloat(toPNorm(mul(0.669692685320417, 0.059834549400923365))))), toProbFloat(protectedDiv(mul(add(protectedSub(91.16142138400603, 0.319795250143737), toBoundedFloat(2.403845468324107)), toProbFloat(add(7.57771026502343, 1.6189634928816672))), toBoundedFloat(protectedSub(toPNorm(2.33849772576853), protectedSub(65.60530884829414, 0.5122993108164792))))), toProbFloat(protectedSub(toPNorm(toPNorm(mul(0.31504361009890536, 1.3202625498936245))), protectedDiv(mul(add(48.20455698591552, 82.2620235436647), mul(48.46110968327609, 32.473106196687304)), toPNorm(toBoundedFloat(0.6276036092027963))))), toProbFloat(protectedDiv(protectedDiv(toProbFloat(toPNorm(68.62137772164792)), mul(toPNorm(52.47548352932008), toBoundedFloat(0.37462550846392895))), protectedSub(toBoundedFloat(protectedSub(6.585412760701315, 9.129491130187873)), protectedSub(protectedDiv(77.64158846016907, 33.83168275131966), mul(0.4817853766706556, 62.13506928889417))))))"
# sample_genome = "FCOS_Head(LazyConvTranspose2d(ResNeXt(ConvNeXt(DenseNet(ConvNeXt(ReLU_2D(MobileNet_V3(IN0, 1, 1)), 0, 1), 1, 0), 2, 1), 0, 2), 28, 7, 3, 9, 8, 7, 0, 3, 1, 2, 1), SGD(0.58, 2, 4.56, 7.02), LinearLR(2.83, 4.50e-06, 2309), 0.78, 0.41, 0.59, 0.48, 0.62, 0, 0.61)"
# print(f"\nSample genome: {sample_genome}", flush=True)

def build_codestral_dataset(use_build_dataset=True, dataset_prefix="mix_dataset", use_preanalysis=True, only_cls_dataset=False):
    """
    Build a dataset using Codestral embeddings from the holy grail CSV file
    Now includes pre-analysis to identify problematic genomes before main processing
    """
    
    print("\n" + "="*80, flush=True)
    print("STARTING CODESTRAL DATASET CREATION", flush=True)
    print("="*80, flush=True)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"use_build_dataset: {use_build_dataset}", flush=True)
    print(f"dataset_prefix: {dataset_prefix}", flush=True)
    print(f"use_preanalysis: {use_preanalysis}", flush=True)
    
    # Parameters for dataset creation
    holy_grail_csv = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_vae_30/out.csv"
    working_dir = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_vae_30"
    outdir = f"/storage/ice-shared/vip-vvk/data/AOT/{USER}/codestral/large_dataset"
    status_dir = f"/storage/ice-shared/vip-vvk/data/AOT/{USER}/codestral/preanalysis_status"

    print("Building raw genome dataset from holy grail CSV...", flush=True)
    print(f"Input file: {holy_grail_csv}", flush=True)
    print(f"Working dir: {working_dir}", flush=True)
    print(f"Output dir: {outdir}", flush=True)
    print(f"Status dir: {status_dir}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(status_dir, exist_ok=True)
    
    if use_build_dataset:
        # Build dataset with raw genomes (no encoding)
        reg_train_set, reg_val_set, cls_train_set, cls_val_set = build_dataset(
            name="codestral_raw",
            infile=holy_grail_csv,
            working_dir=working_dir,
            outdir=outdir,
            metrics='uw_val_epoch_loss,iou_loss,giou_loss,diou_loss,ciou_loss,center_loss,size_loss,obj_loss,precision,recall,f1_score,average_precision',
            exclude=[],
            include_only=None,
            val_ratio=0.3,
            seed=42,
            return_raw_genomes=True
        )
        
        # Save the raw datasets for pre-analysis
        dataset_files = {
            'reg_train': os.path.join(outdir, "codestral_raw_reg_train.pkl"),
            'reg_val': os.path.join(outdir, "codestral_raw_reg_val.pkl"),
            'cls_train': os.path.join(outdir, "codestral_raw_cls_train.pkl"),
            'cls_val': os.path.join(outdir, "codestral_raw_cls_val.pkl")
        }
        
        reg_train_set.to_pickle(dataset_files['reg_train'])
        reg_val_set.to_pickle(dataset_files['reg_val'])
        cls_train_set.to_pickle(dataset_files['cls_train'])
        cls_val_set.to_pickle(dataset_files['cls_val'])
        
    else:
        # Load pre-built raw dataset
        dataset_files = {
            'reg_train': os.path.join(outdir, f"{dataset_prefix}_reg_train.pkl"),
            'reg_val': os.path.join(outdir, f"{dataset_prefix}_reg_val.pkl"),
            'cls_train': os.path.join(outdir, f"{dataset_prefix}_cls_train.pkl"),
            'cls_val': os.path.join(outdir, f"{dataset_prefix}_cls_val.pkl")
        }
        
        reg_train_set = pd.read_pickle(dataset_files['reg_train'])
        reg_val_set = pd.read_pickle(dataset_files['reg_val'])
        cls_train_set = pd.read_pickle(dataset_files['cls_train'])
        cls_val_set = pd.read_pickle(dataset_files['cls_val'])

    print(f"Raw dataset loaded:", flush=True)
    print(f"  Regression train: {len(reg_train_set)} samples", flush=True)
    print(f"  Regression val: {len(reg_val_set)} samples", flush=True)
    print(f"  Classification train: {len(cls_train_set)} samples", flush=True)
    print(f"  Classification val: {len(cls_val_set)} samples", flush=True)

    # Step 1: Pre-analysis phase
    if use_preanalysis:
        print("\n" + "="*60)
        print("PHASE 1: PRE-ANALYSIS - IDENTIFYING PROBLEMATIC GENOMES", flush=True)
        print("="*60)
        
        datasets = {
            'reg_train': (dataset_files['reg_train'], reg_train_set),
            'reg_val': (dataset_files['reg_val'], reg_val_set),
            'cls_train': (dataset_files['cls_train'], cls_train_set),
            'cls_val': (dataset_files['cls_val'], cls_val_set)
        }
        
        # Create mapping of dataset names to dataset files for monitor_preanalyzers
        dataset_file_mapping = {
            dataset_name: dataset_file 
            for dataset_name, (dataset_file, dataset_df) in datasets.items()
        }
        
        # Print dataset sizes
        for dataset_name, (dataset_file, dataset_df) in datasets.items():
            print(f"{dataset_name}: {len(dataset_df)} samples")
        
        # Launch all pre-analyzer jobs simultaneously and monitor them
        print("Launching all pre-analyzer jobs simultaneously...", flush=True)
        success = monitor_preanalyzers(dataset_file_mapping, status_dir)
        
        if not success:
            print("Pre-analysis failed for one or more datasets", flush=True)
            return None
            
        print("Pre-analysis completed for all datasets!", flush=True)
    
    # Step 2: Load failure lists from pre-analysis
    failure_sets = {}
    if use_preanalysis:
        print("\nLoading pre-analysis results...")
        for dataset_name in ['reg_train', 'reg_val', 'cls_train', 'cls_val']:
            failure_file = os.path.join(status_dir, f"{dataset_name}_failures.jsonl")
            failure_sets[dataset_name] = load_failure_list(failure_file)
            print(f"  {dataset_name}: {len(failure_sets[dataset_name])} failed genomes identified")
    else:
        # No pre-analysis, empty failure sets
        for dataset_name in ['reg_train', 'reg_val', 'cls_train', 'cls_val']:
            failure_sets[dataset_name] = set()
    
    # Now process the datasets to generate Codestral embeddings
    print("\nGenerating Codestral embeddings...")
    
    # Initialize Codestral model
    tokenizer, codestral_model, device = initialize_codestral_model()
    
    if tokenizer is None or codestral_model is None:
        print("Failed to initialize Codestral model. Aborting dataset creation.")
        return
    
    def process_dataset_with_codestral(df, dataset_name):
        """Process a dataset by generating Codestral embeddings for each genome"""
        print(f"\nProcessing {dataset_name} ({len(df)} samples)...")
        
        # Get the failure set for this dataset
        dataset_failures = failure_sets.get(dataset_name, set())
        print(f"Will skip {len(dataset_failures)} known problematic genomes")
        
        processed_data = []
        failed_count = 0
        skipped_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
            try:
                # Clear CUDA cache before processing each sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                genome_str = row['str_genome']
                
                # Ensure genome_str is a proper string, not bytes
                if isinstance(genome_str, bytes):
                    genome_str = genome_str.decode('utf-8')
                elif not isinstance(genome_str, str):
                    genome_str = str(genome_str)
                
                # Skip empty or invalid genomes
                if not genome_str or genome_str.strip() == '' or genome_str == 'nan':
                    failed_count += 1
                    print(f"Skipping invalid genome at sample {idx}: {genome_str}")
                    continue
                
                # Check if this genome was identified as problematic during pre-analysis
                genome_hash = get_genome_hash(genome_str)
                if genome_hash in dataset_failures:
                    skipped_count += 1
                    if skipped_count % 100 == 1:  # Print occasionally to show progress
                        print(f"Skipping known problematic genome {idx} (hash: {genome_hash})")
                    continue
                
                # Decode genome to get model representation
                model_dict = codec.decode_genome(genome_str, num_loss_comp)
                model = model_dict['model']
                model.cpu()
                model.eval()
                # Clear CUDA cache after moving model to CPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create text representation
                model_text, extra_vector = create_model_text_representation(model, model_dict)
                
                # Clear CUDA cache before Codestral embedding generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Generate Codestral embedding
                codestral_embeddings = get_codestral_embedding(model_text, tokenizer, codestral_model, device)
                
                if codestral_embeddings is not None:
                    # Create new row with Codestral embedding
                    new_row = row.copy()
                    # Use last token embedding as the primary genome representation
                    new_row['genome'] = np.concatenate([codestral_embeddings['last_token'], extra_vector], axis=0)
                    # new_row['codestral_mean_pooled'] = codestral_embeddings['mean_pooled']
                    # new_row['original_genome'] = genome_str  # Keep original for reference
                    
                    processed_data.append(new_row)
                else:
                    failed_count += 1
                    print(f"Failed to generate embedding for sample {idx}")
                
                # Explicitly delete model to free memory
                del model
                del model_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError as cuda_e:
                failed_count += 1
                print(f"CUDA out of memory for sample {idx}: {cuda_e}")
                # Clear CUDA cache and continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                failed_count += 1
                print(f"Error processing sample {idx}: {e}")
                # Print more details for debugging
                try:
                    genome_str = row['str_genome']
                    print(f"  Genome type: {type(genome_str)}")
                    print(f"  Genome content (first 100 chars): {str(genome_str)[:100]}")
                except:
                    print(f"  Could not inspect genome content")
                
                # Clear CUDA cache in case of error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"Processing complete for {dataset_name}:")
        print(f"  Successfully processed: {len(processed_data)} samples")
        print(f"  Skipped (pre-analysis): {skipped_count}")
        print(f"  Runtime failures: {failed_count}")
        print(f"  Total input samples: {len(df)}")

        final_df = pd.DataFrame(processed_data)
        final_df.drop(columns=['str_genome', 'epoch_num'], inplace=True)
        return final_df

    # Process all datasets
    print("\n" + "="*60)
    print("PROCESSING DATASETS WITH CODESTRAL EMBEDDINGS")
    print("="*60)
    
    if not only_cls_dataset:
        codestral_reg_train = process_dataset_with_codestral(
            reg_train_set, "reg_train"
        )
        codestral_reg_train.to_pickle(os.path.join(outdir, "codestral_reg_train.pkl"))
        print("Saved codestral_reg_train.pkl")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        codestral_reg_val = process_dataset_with_codestral(
            reg_val_set, "reg_val"
        )
        codestral_reg_val.to_pickle(os.path.join(outdir, "codestral_reg_val.pkl"))
        print("Saved codestral_reg_val.pkl")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    codestral_cls_train = process_dataset_with_codestral(
        cls_train_set, "cls_train"
    )
    codestral_cls_train.to_pickle(os.path.join(outdir, "codestral_cls_train.pkl"))
    print("Saved codestral_cls_train.pkl")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    codestral_cls_val = process_dataset_with_codestral(
        cls_val_set, "cls_val"
    )
    codestral_cls_val.to_pickle(os.path.join(outdir, "codestral_cls_val.pkl"))
    print("Saved codestral_cls_val.pkl")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save the processed datasets
    
    print(f"\nCodestral datasets saved to {outdir}:")
    if not only_cls_dataset:
        print(f"  codestral_reg_train.pkl: {len(codestral_reg_train)} samples")
        print(f"  codestral_reg_val.pkl: {len(codestral_reg_val)} samples")
    print(f"  codestral_cls_train.pkl: {len(codestral_cls_train)} samples")
    print(f"  codestral_cls_val.pkl: {len(codestral_cls_val)} samples")
    
    # Print embedding statistics
    if len(codestral_cls_train) > 0:
        sample_embedding = codestral_cls_train.iloc[0]['genome']
        print(f"\nEmbedding statistics:")
        print(f"  Embedding dimension: {sample_embedding.shape}")
        print(f"  Sample embedding mean: {np.mean(sample_embedding):.4f}")
        print(f"  Sample embedding std: {np.std(sample_embedding):.4f}")
    
    if not only_cls_dataset:
        return codestral_reg_train, codestral_reg_val, codestral_cls_train, codestral_cls_val
    return codestral_cls_train, codestral_cls_val


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Codestral Dataset Creation and Model Inspection')
    parser.add_argument('--mode', choices=['inspect', 'dataset'], default='inspect',
                        help='Mode: inspect a single model or build entire dataset')
    parser.add_argument('--only_cls_dataset', action='store_true',
                        help='If building dataset, only build classification dataset')
    
    args = parser.parse_args()
    
    if args.mode == 'inspect':
        # Original functionality - decode and inspect the model
        decoded_model, decoded_model_dict, codestral_embeddings = decode_and_inspect_model(sample_genome, codec, num_loss_comp)
    elif args.mode == 'dataset':
        # New functionality - build Codestral dataset
        print("Building Codestral dataset...")
        codestral_datasets = build_codestral_dataset(use_build_dataset=False,
                                                     dataset_prefix="mix_dataset", use_preanalysis=True,
                                                     only_cls_dataset=args.only_cls_dataset)
        print("Dataset creation completed!")

