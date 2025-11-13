#!/usr/bin/env python3
"""
Codestral Pre-analyzer: Disposable process for testing genome decoding
This script runs in a separate process to test which genomes can be successfully decoded
without crashing the main process due to CUDA illegal memory access errors.
"""

import os
import sys
import torch
import pandas as pd
import argparse
import traceback
import time
from pathlib import Path

import toml
from codec import Codec

# Load configuration
cwd = os.path.dirname(os.getcwd())
cfg = toml.load(os.path.join(cwd, "conf.toml"))
genome_encoding_strat = cfg["codec"]['genome_encoding_strat']
num_classes = cfg["model"]['num_classes']
num_loss_comp = cfg["model"]['num_loss_components']

def update_status_file(status_file, current_idx, status, genome_hash=None, error_msg=None):
    """Update the status file with current progress"""
    status_data = {
        'current_idx': current_idx,
        'status': status,  # 'processing', 'success', 'failed', 'completed'
        'timestamp': time.time()
    }
    
    if genome_hash:
        status_data['genome_hash'] = genome_hash
    if error_msg:
        status_data['error_msg'] = error_msg
    
    # Write to temp file first, then rename for atomic update
    temp_file = f"{status_file}.tmp"
    with open(temp_file, 'w') as f:
        import json
        json.dump(status_data, f)
    
    # Atomic rename
    os.rename(temp_file, status_file)

def load_status_file(status_file):
    """Load the current status from file"""
    if not os.path.exists(status_file):
        return {'current_idx': 0, 'status': 'starting'}
    
    try:
        with open(status_file, 'r') as f:
            import json
            return json.load(f)
    except:
        # If file is corrupted, start from beginning
        return {'current_idx': 0, 'status': 'starting'}

def update_failure_file(failure_file, genome_hash, error_msg):
    """Add a failed genome to the failure tracking file"""
    failure_entry = {
        'genome_hash': genome_hash,
        'error_msg': error_msg,
        'timestamp': time.time()
    }
    
    # Append to failure file
    with open(failure_file, 'a') as f:
        import json
        f.write(json.dumps(failure_entry) + '\n')

def get_genome_hash(genome_str):
    """Generate a simple hash for the genome string"""
    import hashlib
    return hashlib.md5(genome_str.encode()).hexdigest()[:10]

def test_genome_decode(genome_str, codec):
    """
    Test if a genome can be successfully decoded without crashing
    Returns: (success: bool, error_msg: str or None)
    """
    try:
        print(f"    Starting decode test...")
        
        # Clear CUDA cache before attempting decode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"    Calling codec.decode_genome...")
        # Attempt to decode the genome
        model_dict = codec.decode_genome(genome_str, num_loss_comp)
        print(f"    Decode successful, got model_dict with keys: {model_dict.keys()}")
        
        model = model_dict['model']
        print(f"    Got model: {type(model)}")
        
        # Move to CPU and clean up immediately
        model.cpu()
        del model
        del model_dict
        
        # Clear CUDA cache after successful decode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"    Decode test completed successfully")
        return True, None
        
    except RuntimeError as e:
        error_msg = str(e)
        print(f"    RuntimeError: {error_msg}")
        traceback.print_exc()
        # Check for CUDA illegal memory access or other GPU errors
        if "illegal memory access" in error_msg or "CUDA error" in error_msg:
            return False, f"CUDA_ERROR: {error_msg}"
        else:
            return False, f"RUNTIME_ERROR: {error_msg}"
    except Exception as e:
        error_msg = str(e)
        print(f"    General Exception: {type(e).__name__}: {error_msg}")
        print(f"    Exception details:")
        traceback.print_exc()
        return False, f"GENERAL_ERROR: {error_msg}"

def main():
    parser = argparse.ArgumentParser(description='Pre-analyze genomes for Codestral processing')
    parser.add_argument('--dataset_file', required=True, help='Path to the dataset pickle file')
    parser.add_argument('--dataset_name', required=True, help='Name of the dataset (for logging)')
    parser.add_argument('--status_dir', required=True, help='Directory to store status files')
    parser.add_argument('--start_idx', type=int, default=0, help='Index to start processing from')
    
    args = parser.parse_args()
    
    print(f"Pre-analyzer starting for {args.dataset_name}")
    print(f"Dataset file: {args.dataset_file}")
    print(f"Status directory: {args.status_dir}")
    print(f"Starting from index: {args.start_idx}")
    
    # Ensure status directory exists
    os.makedirs(args.status_dir, exist_ok=True)
    
    # Setup status and failure files
    status_file = os.path.join(args.status_dir, f"{args.dataset_name}_status.json")
    failure_file = os.path.join(args.status_dir, f"{args.dataset_name}_failures.jsonl")
    
    # Initialize codec
    print("Initializing Codec...")
    print(f"  num_classes: {num_classes}")
    print(f"  genome_encoding_strat: {genome_encoding_strat}")
    print(f"  num_loss_comp: {num_loss_comp}")
    
    try:
        codec = Codec(num_classes, genome_encoding_strat=genome_encoding_strat)
        print("Codec initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize codec: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_file}...")
    df = pd.read_pickle(args.dataset_file)
    print(f"Dataset loaded: {len(df)} samples")
    
    # Load current status
    status = load_status_file(status_file)
    start_idx = max(args.start_idx, status.get('current_idx', 0))
    
    print(f"Starting pre-analysis from index {start_idx}")
    
    # Process each genome starting from the specified index
    for idx in range(start_idx, len(df)):
        try:
            # Update status to show we're processing this index
            update_status_file(status_file, idx, 'processing')
            
            row = df.iloc[idx]
            genome_str = row['str_genome']
            
            # Ensure genome_str is a proper string
            if isinstance(genome_str, bytes):
                genome_str = genome_str.decode('utf-8')
            elif not isinstance(genome_str, str):
                genome_str = str(genome_str)
            
            # Skip empty or invalid genomes
            if not genome_str or genome_str.strip() == '' or genome_str == 'nan':
                print(f"Skipping invalid genome at index {idx}")
                update_status_file(status_file, idx + 1, 'success')
                continue
            
            genome_hash = get_genome_hash(genome_str)
            print(f"Testing genome {idx}/{len(df)} (hash: {genome_hash})")
            print(f"  Genome type: {type(genome_str)}")
            print(f"  Genome length: {len(genome_str)}")
            print(f"  Genome preview: {genome_str[:100]}...")
            
            # Test if the genome can be decoded
            success, error_msg = test_genome_decode(genome_str, codec)
            
            if success:
                print(f"  ✓ Success")
                update_status_file(status_file, idx + 1, 'success', genome_hash)
            else:
                print(f"  ✗ Failed: {error_msg}")
                update_status_file(status_file, idx + 1, 'failed', genome_hash, error_msg)
                update_failure_file(failure_file, genome_hash, error_msg)
                
                # If it's a CUDA error, we might need to exit and let the parent restart us
                if "CUDA_ERROR" in error_msg and "illegal memory access" in error_msg:
                    print(f"FATAL: Illegal memory access detected. Process needs restart.")
                    update_status_file(status_file, idx + 1, 'fatal_error', genome_hash, error_msg)
                    sys.exit(1)  # Exit with error code
        
        except Exception as e:
            error_msg = f"Unexpected error processing index {idx}: {str(e)}"
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            
            genome_hash = "unknown"
            try:
                genome_hash = get_genome_hash(str(df.iloc[idx]['str_genome']))
            except:
                pass
            
            update_status_file(status_file, idx + 1, 'error', genome_hash, error_msg)
            update_failure_file(failure_file, genome_hash, error_msg)
            
            # Continue to next genome
            continue
    
    # Mark as completed
    update_status_file(status_file, len(df), 'completed')
    print(f"Pre-analysis completed for {args.dataset_name}")
    print(f"Processed {len(df)} genomes total")

if __name__ == "__main__":
    main()