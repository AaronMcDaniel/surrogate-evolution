"""
t-SNE Visualization Script for Codestral Genome Embeddings

This script loads genome embeddings from a pickle file and creates
t-SNE visualizations to explore the embedding space.
"""
import sys, types, numpy as np
    




import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import argparse
import os
from tqdm import tqdm

def load_dataset(file_path):
    """Load the dataset from pickle file"""
    print(f"Loading dataset from {file_path}...")
    df = pd.read_pickle(file_path)
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    return df

def extract_genome_vectors(df):
    """Extract genome vectors from the dataframe"""
    print("Extracting genome vectors...")
    
    # Check if 'genome' column exists
    if 'genome' not in df.columns:
        raise ValueError(f"'genome' column not found. Available columns: {df.columns.tolist()}")
    
    # Extract genome vectors
    genome_vectors = []
    valid_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing genomes"):
        genome = row['genome']
        
        # Handle different possible formats
        if isinstance(genome, np.ndarray):
            genome_vectors.append(genome.flatten())
            valid_rows.append(row)
        elif isinstance(genome, (list, tuple)):
            genome_vectors.append(np.array(genome).flatten())
            valid_rows.append(row)
        elif isinstance(genome, str):
            # If it's a string, it might be the raw genome (not encoded yet)
            print(f"Warning: Genome at index {idx} is a string, not a vector. Skipping.")
        else:
            print(f"Warning: Unknown genome format at index {idx}: {type(genome)}")
    
    if not genome_vectors:
        raise ValueError("No valid genome vectors found in the dataset")
    
    genome_matrix = np.vstack(genome_vectors)
    # Create a new DataFrame from valid rows with reset index
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    
    print(f"Genome matrix shape: {genome_matrix.shape}")
    print(f"Valid samples: {len(valid_rows)}/{len(df)}")
    
    return genome_matrix, valid_df

def compute_tsne(genome_matrix, perplexity=30, n_iter=1000, random_state=42, n_components=2):
    """Compute t-SNE embedding"""
    print(f"Computing t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    
    # Standardize the data
    print("Standardizing features...")
    scaler = StandardScaler()
    genome_matrix_scaled = scaler.fit_transform(genome_matrix)
    
    # Compute t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1
    )
    
    tsne_embedding = tsne.fit_transform(genome_matrix_scaled)
    print(f"t-SNE embedding shape: {tsne_embedding.shape}")
    
    return tsne_embedding

def create_visualizations(tsne_embedding, df, output_dir, output_prefix):
    """Create various t-SNE visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 10)
    
    # 1. Basic scatter plot
    print("Creating basic scatter plot...")
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                alpha=0.6, s=30, c='steelblue', edgecolors='none')
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.title('t-SNE Visualization of Genome Embeddings', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_prefix}_tsne_basic.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, f'{output_prefix}_tsne_basic.png')}")
    
    # 2. Density plot
    print("Creating density plot...")
    plt.figure(figsize=(12, 10))
    plt.hexbin(tsne_embedding[:, 0], tsne_embedding[:, 1], 
               gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(label='Point Density')
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.title('t-SNE Density Plot of Genome Embeddings', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_prefix}_tsne_density.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, f'{output_prefix}_tsne_density.png')}")
    
    # 3. Color by performance metrics if available
    metric_columns = [col for col in df.columns if col not in ['genome', 'hash', 'str_genome', 'epoch_num']]
    if 'average_precision' in df.columns:
        metric_columns.remove('average_precision')
        metric_columns.insert(0, 'average_precision')
    if 'ciou_loss' in df.columns:
        metric_columns.remove('ciou_loss')
        metric_columns.insert(0, 'ciou_loss')
    
    #add reverse mapping from tsne vectors to hash values
    tsne_to_hash = {tuple(tsne_embedding[i]): df.loc[i, 'hash'] for i in range(len(df))}
    #store as csv
    hash_mapping_file = os.path.join(output_dir, f'{output_prefix}_tsne_to_hash_mapping.csv')
    with open(hash_mapping_file, 'w') as f:
        f.write("TSNE_Component_1,TSNE_Component_2,Hash\n")
        for i in range(len(df)):
            f.write(f"{(tsne_embedding[i,0],tsne_embedding[i,1])},{df.loc[i,'hash']}\n")
    print(f"Saved: {hash_mapping_file}")

    
    if metric_columns:
        print(f"Creating visualizations colored by metrics: {metric_columns}")
        
        for metric in metric_columns[:5]:  # Limit to first 5 metrics to avoid too many plots
            try:
                values = df[metric].values
                
                # Skip if not numeric
                if not np.issubdtype(values.dtype, np.number):
                    continue
                
                # Skip if all NaN
                if np.all(np.isnan(values)):
                    continue
                
                plt.figure(figsize=(12, 10))
                scatter = plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                                     c=values, alpha=0.6, s=30, cmap='coolwarm', 
                                     edgecolors='none')
                plt.colorbar(scatter, label=metric)
                plt.xlabel('t-SNE Component 1', fontsize=14)
                plt.ylabel('t-SNE Component 2', fontsize=14)
                plt.title(f't-SNE Visualization colored by {metric}', fontsize=16)
                plt.tight_layout()
                
                safe_metric_name = metric.replace('/', '_').replace(' ', '_')
                plt.savefig(os.path.join(output_dir, f'{output_prefix}_tsne_by_{safe_metric_name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved: {os.path.join(output_dir, f'{output_prefix}_tsne_by_{safe_metric_name}.png')}")
                
            except Exception as e:
                print(f"Warning: Could not create plot for {metric}: {e}")
    
    # 4. Create interactive HTML if plotly is available
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        print("Creating interactive plot...")
        
        # Prepare data for plotly
        plot_df = pd.DataFrame({
            'TSNE-1': tsne_embedding[:, 0],
            'TSNE-2': tsne_embedding[:, 1]
        })
        
        # Add first metric if available
        if metric_columns:
            first_metric = metric_columns[0]
            if np.issubdtype(df[first_metric].dtype, np.number):
                plot_df[first_metric] = df[first_metric].values
        
        fig = px.scatter(plot_df, x='TSNE-1', y='TSNE-2',
                        color=first_metric if metric_columns and first_metric in plot_df else None,
                        title='Interactive t-SNE Visualization',
                        width=1000, height=800)
        
        fig.update_traces(marker=dict(size=5, opacity=0.6))
        fig.write_html(os.path.join(output_dir, 'tsne_interactive.html'))
        print(f"Saved: {os.path.join(output_dir, 'tsne_interactive.html')}")
        
    except ImportError:
        print("Plotly not available, skipping interactive plot")
    except Exception as e:
        print(f"Warning: Could not create interactive plot: {e}")
    
    print(f"\nAll visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='t-SNE Visualization of Genome Embeddings')
    parser.add_argument('--input', type=str, 
                       default='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/codestral_raw_reg_train.pkl',
                       help='Path to input pickle file')
    parser.add_argument('--output_dir', type=str,
                       default='/storage/ice-shared/vip-vvk/data/AOT/mgullapalli6/codestral/tsne_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter (default: 30)')
    parser.add_argument('--n_iter', type=int, default=1000,
                       help='Number of t-SNE iterations (default: 1000)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (for faster testing)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--output_prefix', type=str, default="codestral",
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    print("="*60)
    print("t-SNE Visualization of Genome Embeddings")
    print("="*60)
    
    # Load dataset
    df = load_dataset(args.input)
    
    # Subsample if requested
    if args.max_samples and len(df) > args.max_samples:
        print(f"Subsampling to {args.max_samples} samples...")
        df = df.sample(n=args.max_samples, random_state=args.random_state)
    
    # Extract genome vectors
    genome_matrix, valid_df = extract_genome_vectors(df)
    
    # Compute t-SNE
    tsne_embedding = compute_tsne(
        genome_matrix, 
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=args.random_state
    )
    
    # Create visualizations
    create_visualizations(tsne_embedding, valid_df, args.output_dir, args.output_prefix)
    
    # Save t-SNE embeddings
    embedding_file = os.path.join(args.output_dir, f'{args.output_prefix}_tsne_embeddings.npz')
    np.savez(embedding_file, 
             tsne_embedding=tsne_embedding)
    print(f"\nt-SNE embeddings saved to: {embedding_file}")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

if __name__ == "__main__":
    main()
