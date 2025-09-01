# Surrogate-Assisted Neural Architecture Search (SA-NAS)

SA-NAS is an AutoML technique aimed at automatically discovering optimal neural network architectures for specific tasks within a predefined search space. Its core purpose is to construct efficient and powerful neural networks without requiring specialized domain knowledge for manual design, leading to substantial reductions in computational costs. This project specifically addresses Object Detection-Neural Architecture Search (NAS), utilizing Airborne Object Tracking (AOT) datasets.

## Quick Setup

### 1. Environment Setup
```bash
# Load modules
module load anaconda3/2023.03
module load cuda/11.8.0

# Create and activate conda environment
conda create -n nas python==3.12.2
conda activate nas

# Install dependencies
pip install git+https://github.com/jasonzutty/deap.git
pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1
pip install scikit-learn==1.5.0 opencv-python==4.10.0.84
pip install pillow==10.2.0 scipy==1.13.1 seaborn==0.13.2 matplotlib==3.8.4
pip install pandas==2.2.1 tqdm==4.66.4 toml==0.10.2 future==1.0.0 numpy==1.26.4
```

### 2. Repository Setup
```bash
# Clone and setup repository
cd ~/scratch
git clone https://github.com/AaronMcDaniel/surrogate-evolution.git
cd surrogate-evolution
git checkout pranav_evo_runs

# Create workspace directories (replace 'your_username' with your GT username)
mkdir -p /storage/ice-shared/vip-vvk/data/AOT/your_username/evolution_logs
mkdir -p /storage/ice-shared/vip-vvk/data/AOT/your_username/output

# Setup symlinks
mv ~/.conda ~/scratch 2>/dev/null || true
ln -s ~/scratch/.conda ~/.conda
rm -fr ~/.cache
mkdir ~/scratch/.cache
ln -s ~/scratch/.cache ~/.cache
```

### 3. Configure Files
Edit these files and replace usernames with your GT username:
- `main.job` lines 7-8: Update log paths to your directory
- `pipeline.py`: Change "pco" to "nas"
- `conf.toml` line 101: Set pretrained_dir path

### 4. Run Evolution
```bash
sbatch main.job -o "/storage/ice-shared/vip-vvk/data/AOT/your_username/output" -conf conf.toml -n 30 -e nas -s seeds.txt -r
```

## Core Components and Methodology

The methodology is built upon three integrated components: a search space adapted for Deep Neural Networks (DNNs), an evolutionary search strategy, and a performance evaluation mechanism.

### 1. Evolutionary Algorithm

The system primarily employs a strongly-typed Genetic Programming (GP) algorithm, implemented using the DEAP (Distributed Evolutionary Algorithms in Python) package.

**Optimization Strategies:**
- NSGA-III is used for multi-objective optimization, particularly for Super-Resolution Image Restoration (SRIR) tasks, where objectives include maximizing Peak-Signal-to-Noise Ratio (PSNR) and minimizing Floating-Point Operations (FLOPs) and learnable parameters.
- NSGA2 is utilized for operations such as downselection and parent selection within the evolutionary pipeline.

**Objectives:**
- For SRIR, the objectives are minimizing FLOPs, minimizing learnable parameters, and maximizing PSNR.
- For Object Detection, objectives include minimizing Clou Loss, maximizing Average Precision, and minimizing UW Validation Loss.

**Genetic Operators:** Mating (crossover) and mutation operators introduce architectural changes. Type-fixed mate/mutation operators have been explored to reduce hyperparameter bloat.

### 2. Surrogate Models

Surrogate models, or meta-models, are lightweight approximation models that predict genome performance, thereby replacing expensive real evaluations and accelerating convergence to optimal solutions.

**Models Implemented/Under Development:**
- MLP (Multi-Layer Perceptron).
- KAN (Kolmogorov-Arnold Network), identified as an alternative to MLP that handles high-dimensional data effectively.
- GNN (Graph Neural Network), currently under development.
- An XGBoost-based surrogate model has been proposed for SRIR tasks, demonstrating significant speed improvements.
- A Transformer-based surrogate model has been implemented and shows outperformance over MLP and KAN on at least one dataset.

**Input Features:** Surrogates use features derived from architectural configurations, including operation types, kernel sizes, repetition patterns, channel numbers, tree characteristics (nodes, depth), parents' fitness, and predicted class labels from subsets of training instances.

**Two-Stage Approach:** The surrogate solution typically involves two stages: first, classifying failed genomes, and then regressing genome performance metrics.

**Training and Integration:** Surrogate models are trained on architecture-performance pairs. This training dataset is iteratively updated with data from new generations (online learning). Faster surrogate training dispatches GPU jobs and caches results.

### 3. Encoding Strategies

Different encoding methods represent neural network architectures for both evolutionary search and surrogate model training.

**Genome Encoding (for Evolutionary Algorithm):**
- Tree-based representation is used for genome encoding, employing syntax trees of terminals (leaves) and non-terminals (nodes).
- Linear GP representation is an alternative.

**Surrogate Encoding (for Surrogate Models):**
- **String2Vec:** Parses layers and arguments from a string representation, transforming them into a numerical vector (one-hot encoding for layer/block types, actual values for hyperparameters), then flattening to a 1D vector.
- **Strongly-Typed String2Vec:** An extension using "buckets" for specific data types, aiming to capture all node parameters. Initial experiments showed this performed worse than original String2Vec, possibly due to its larger size (2146 vs. 1021 features) and many blank values.
- **Graph-based Encoding:** Converts PyTorch models into a computation graph, one-hot encodes modules by name, and converts to a torch-geometric data object. A refined version incorporates module-specific hyperparameters into node features.
- **Autoencoder-Based Encoding:** Used in EAEPSO, it compresses variable-length discrete integer block vectors into fixed-length continuous decimal latent vectors, making them suitable for Particle Swarm Optimization (PSO). A Variational Autoencoder (VAE) preprocessing step has reduced encoding dimensionality and improved learnability.

### 4. Fitness Evaluation

The fitness of neural network designs is determined by training and testing, with top-performing architectures advancing.

**Computational Cost:** Fully training each candidate architecture is computationally intensive and time-consuming. Surrogates mitigate this cost.

**Real Fitness Function:** For classification, accuracy is typically the real fitness function, often employing stratified K-fold cross-validation for robustness.

**Dynamic Hierarchical Fitness Evaluation:** In PSO-based ENAS, a progressive approach starts with a small subset of training data and gradually increases the data scale to improve rank consistency.

## Operational Pipeline

The high-level data pipeline for SA-NAS follows an iterative process:

1. **Initialization:** Generating an initial population of individuals from seeds or random generation.
2. **Evaluation:** Calculating real fitness values for a subset of individuals to set initial fitnesses.
3. **Train Surrogate:** Building and training the surrogate model using data collected from previous generations and pre-training data.
4. **Update Elite Pool & Hall of Fame:** Maintaining a record of top-performing individuals.
5. **Select Parents:** Choosing individuals from the population and elite pool using NSGA2.
6. **Overpopulation:** Generating new individuals through crossover and mutation.
7. **Downselection:** Reducing the population size using surrogate inferences, potentially combined with dynamic NSGA2 and random selection.
8. **Repeat:** These steps iterate until termination criteria are met.

## Simulated Surrogate Injection (SSI)

Simulated Surrogate Injection (SSI) is an advanced evolutionary loop designed to intensively leverage the surrogate model after the overpopulation step, aiming to optimize search efficiency.

**Process:** SSI begins by using existing (true) fitness values for initial performance understanding. It then classifies and removes invalid individuals using surrogate classification, and assigns objective fitnesses to valid individuals via surrogate regression. Parents are selected from this valid pool to reproduce an "unsustainable population" (e.g., 2000 individuals). This overpopulated group is then downselected using NSGA2 to a reserved partition size, and the process repeats for a set number of iterations.

**Robustness:** Improvements ensure SSI starts with as much ground truth information as possible and has better recovery plans for poor surrogate searches, including reserving a portion of the final population for individuals from the normal evolutionary loop to ensure continuous exploration.

**Efficiency:** Faster surrogate training is achieved by dispatching GPU jobs and caching results, allowing the main CPU job to poll for completion.

**Integration:** Type-fixed mutation operators can be integrated and run within SSI-enabled runs.

## Technologies and Environment

**Programming Language & Libraries:** Python, with the DEAP package for evolutionary algorithms, and PyTorch for defining and decoding neural network models.

**Compute Environment:** Experiments are conducted on compute clusters, specifically mentioning GTRI's ICEHAMMER cluster and PACE-ICE. Slurm is used for job scheduling, managing job dependencies and submission scripts.

**Reproducibility:** Python and NumPy random number generators are seeded to ensure consistent results across runs.

## Current Status & Future Work

Ongoing and future work focuses on:

- **Expanding Surrogate Capabilities:** Further development of GNN models and graph embedding techniques, and exploring transformer-based surrogates.
- **Improving Encoding:** Developing encoding schemes that incorporate more information about the genome, such as hyperparameters and layer attributes, and adapting GNN architectures.
- **Data Augmentation:** Investigating data augmentation techniques and unsupervised learning to increase dataset diversity and size.
- **Pipeline Refinements:** Continuous improvements to evolutionary loop approaches, including further optimization and analysis of SSI implementations.
- **Publication:** A goal to publish research findings.

## Authors

Contributors:
- Aaron McDaniel
- Ethan Harpster
- Tristan Thakur
- Vineet Kulkarni 

## Project status
Ongoing
