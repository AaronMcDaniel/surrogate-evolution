"""
t-SNE Visualization Script for Codestral Genome Embeddings

This script loads genome embeddings from a pickle file and creates
t-SNE visualizations to explore the embedding space.
"""
import math
import sys, types, numpy as np
    




import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import argparse
import os
from tqdm import tqdm
from tree_simplifier import simplify_tree
from  surrogates.surrogate import Surrogate
import torch
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
    if 'surrogate_ciou_loss_error' in df.columns:
        metric_columns.remove('surrogate_ciou_loss_error')
        metric_columns.insert(0, 'surrogate_ciou_loss_error')
    if 'surrogate_average_precision_error' in df.columns:
        metric_columns.remove('surrogate_average_precision_error')
        metric_columns.insert(0, 'surrogate_average_precision_error')
    
    #add reverse mapping from tsne vectors to hash values
    tsne_to_hash = {tuple(tsne_embedding[i]): df.loc[i, 'hash'] for i in range(len(df))}
    #store as csv
    hash_mapping_file = os.path.join(output_dir, f'{output_prefix}_tsne_reverse_mapping.csv')
    with open(hash_mapping_file, 'w') as f:
        f.write("TSNE_Component_1,TSNE_Component_2,Hash\n")
        for i in range(len(df)):
            f.write(f"{(float(tsne_embedding[i,0]),float(tsne_embedding[i,1]))},{df.loc[i,'hash']}\n")
    print(f"Saved: {hash_mapping_file}")

    
    if metric_columns:
        print(f"Creating visualizations colored by metrics: {metric_columns}")
        
        for metric in metric_columns[:7]:  # Limit to first 7 metrics to avoid too many plots
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


#create a function that will classify the tsne mapping without supervision and will find number of clusters and will return a dictionary of the found cluster's hash values using the tsne_to_hash mapping.csv file
#use HDBSCAN
def classify_tsne(tsne_embedding, output_dir, output_prefix, inputhash):
    from sklearn.cluster import HDBSCAN
   
    #load tsne_to hashfrom csv
    tsne_to_hash = {}
    tsne_to_genome = {}
    hash_mapping_file = os.path.join(output_dir, f'{output_prefix}_tsne_reverse_mapping.csv')
    with open(hash_mapping_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            if(len(line.strip().split(',')) < 3):
                continue
            # rewrite the following line since tsne_comp is a tuple and hash_value is a string
            tsne_comp1, tsne_comp2, hash_value = line.strip().split(',')
            tsne_comp1 = float(tsne_comp1[1:])
            tsne_comp2 = float(tsne_comp2[:-1])
            tsne_tuple = tuple([tsne_comp1, tsne_comp2])
            tsne_to_hash[tsne_tuple] = hash_value
    # Final clustering with optimal number of clusters
    clusterer = HDBSCAN(min_cluster_size=5)
    cluster_labels = clusterer.fit_predict(tsne_embedding)
    #print(tsne_to_hash)
    # Create a mapping of cluster labels to hash values
    cluster_to_hashes = {}
    cluster_to_genomes = {}
    for idx, label in enumerate(cluster_labels):
        if label not in cluster_to_hashes:
            cluster_to_hashes[label] = []
        a = tuple(float(x) for x in tsne_embedding[idx])
        cluster_to_hashes[label].append(tsne_to_hash[a])

    #store the cluster to hashes mapping as a csv
    hash_mapping_file = os.path.join(output_dir, f'{output_prefix}_tsne_clusters_numbers.txt')
    with open(hash_mapping_file, 'w') as f:
        f.write("Cluster_Label,Hash_Values\n")
        for cluster_label, hash_values in cluster_to_hashes.items():
            f.write(f"{cluster_label},{';'.join(hash_values)}\n")
    #using --inputhash argument, create a mapping of cluster labels to genome strings
    hash_to_genome = {}
    input_df = pd.read_csv(inputhash)
    for idx, row in input_df.iterrows():
        hash_value = row['hash']
        genome_str = row['genome']
        # use tree simplifier to simplify genome string
        simplified_genome_str = simplify_tree(genome_str)
        hash_to_genome[hash_value] = simplified_genome_str
    for cluster_label, hash_values in cluster_to_hashes.items():
        if cluster_label not in cluster_to_genomes:
            cluster_to_genomes[cluster_label] = []
        for hash_value in hash_values:
            if hash_value in hash_to_genome:
                cluster_to_genomes[cluster_label].append(hash_to_genome[hash_value])
    #store the cluster to genomes mapping as a txt file
    hash_mapping_file = os.path.join(output_dir, f'{output_prefix}_tsne_clusters_genomes.txt')
    with open(hash_mapping_file, 'w') as f:
        f.write("Cluster_Label,Genome_Strings\n")
        for cluster_label, genome_strings in cluster_to_genomes.items():
            f.write(f"{cluster_label}:\n{'\n'.join(genome_strings)}\n\n")
    print(f"Saved: {hash_mapping_file}")
    
    # Create visualization with cluster labels
    print("Creating cluster visualization...")
    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                         c=cluster_labels, alpha=0.6, s=30, cmap='tab20', 
                         edgecolors='none')
    plt.colorbar(scatter, label='Cluster Label')
    
    # Add cluster ID annotations at cluster centroids
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        cluster_mask = cluster_labels == label
        cluster_points = tsne_embedding[cluster_mask]
        centroid = cluster_points.mean(axis=0)
        plt.annotate(str(label), 
                    xy=centroid, 
                    fontsize=12, 
                    fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.7))
    
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.title('t-SNE Visualization with Cluster Labels', fontsize=16)
    plt.tight_layout()
    cluster_viz_file = os.path.join(output_dir, f'{output_prefix}_tsne_clusters.png')
    plt.savefig(cluster_viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cluster_viz_file}")
    createHistogram(hash_mapping_file, output_dir, output_prefix)
    return cluster_to_hashes    
#ctreates histograms for every cluster
def createHistogram(infile, output_dir, output_prefix):
    "Read the genomes.csv file and create a histogram of the primitive counts"
    heads = {"RetinaNet_Head":0, "FPN_Head":0, "SSD_Head":0, "YOLOv3_Head":0, "FasterRCNN_Head":0}
    layers={}
    optimizers={}
    learning_rate_adapters={}
    clusters={}
    clusterNow = -100
    with open(infile, 'r') as f:
        next(f)  # skip header
        for line in f:
            #if line in cluster header add heads, layers, optimizers, learning_rate_adapters to an item in clusters, and setclusternow to that cluster number
            if len(line.strip().split(',')) < 10 and len(line.strip()) > 0 and ':' in line:
                print(clusters[clusterNow] if clusterNow != -100 else "")
                clusterNow = int(line.strip().replace(':', ''))
                clusters[clusterNow] = {"heads":{}, "layers":{}, "optimizers":{}, "learning_rate_adapters":{}}
                continue
            

            elif len(line.strip().split(',')) ==0:
                continue
            #else parse the genome line by splitting top level genome information i.e. (a,b,c),d,(e,(f,g)) => ['(a,b,c)', 'd', '(e,(f,g))']
            genome_str = []
            current_item = ''
            paren_count = 0
            for char in line.strip():
                if char == '(':
                    paren_count += 1
                    current_item += char
                elif char == ')':
                    paren_count -= 1
                    current_item += char
                elif char == ',' and paren_count == 1:
                    genome_str.append(current_item)
                    current_item = ''
                else:
                    current_item += char
            if current_item:
                genome_str.append(current_item)

            #tokenize each item in genome string a list of strings
            for item in genome_str[0:3]:
                #ANY set of contigous alphabetic, "2D" and _ charcters should be a token
                tokens = []
                current_token = ''
                for char in item:
                    if char.isalpha() or char == '_' or (char.isdigit() and current_token):
                        current_token += char
                    else:
                        if current_token:
                            tokens.append(current_token)
                            current_token = ''
                if current_token:
                    tokens.append(current_token)
                #if item in genome_str[0] if contains head add to heads, else add to layers if in str[1] add to optimizerds, else add to learning rate adapters
                for token in tokens:
                    if 'Head' in token and genome_str.index(item) == 0:
                        heads[token] = heads.get(token, 0) + 1
                        if clusterNow != -100:
                            clusters[clusterNow]["heads"][token] = clusters[clusterNow]["heads"].get(token, 0) + 1
                    elif(genome_str.index(item) == 0 and token != "IN0"):
                        layers[token] = layers.get(token, 0) + 1
                        if clusterNow != -100:
                            clusters[clusterNow]["layers"][token] = clusters[clusterNow]["layers"].get(token, 0) + 1
                    elif(genome_str.index(item) == 1 and token != "True" and token != "False"):
                        optimizers[token] = optimizers.get(token, 0) + 1
                        if clusterNow != -100:
                            clusters[clusterNow]["optimizers"][token] = clusters[clusterNow]["optimizers"].get(token, 0) + 1
                    elif(genome_str.index(item) == 2):
                        learning_rate_adapters[token] = learning_rate_adapters.get(token, 0) + 1
                        if clusterNow != -100:
                            clusters[clusterNow]["learning_rate_adapters"][token] = clusters[clusterNow]["learning_rate_adapters"].get(token, 0) + 1
    #create histograms for heads, layers, optimizers, learning_rate_adapters
    def plot_histogram(data_dict, title, filename):
        plt.figure(figsize=(10, 6))
        items = list(data_dict.items())
        items.sort(key=lambda x: x[1], reverse=True)
        keys, values = zip(*items)
        plt.bar(keys, values, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Primitive', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        return plt
    #create a directory fo plots for each cluster
    cluster_plot_dir = os.path.join(output_dir, f'{output_prefix}_cluster_plots')
    os.makedirs(cluster_plot_dir, exist_ok=True)
    #create a subdirectory for each cluster
    for cluster_label, primitives in clusters.items():
        heads_plot = plot_histogram(primitives["heads"], f'Cluster {cluster_label} - Heads Distribution', f'cluster_{cluster_label}_heads_histogram.png')
        layers_plot = plot_histogram(primitives["layers"], f'Cluster {cluster_label} - Layers Distribution', f'cluster_{cluster_label}_layers_histogram.png')
        optimizers_plot = plot_histogram(primitives["optimizers"], f'Cluster {cluster_label} - Optimizers Distribution', f'cluster_{cluster_label}_optimizers_histogram.png')
        lra_plot = plot_histogram(primitives["learning_rate_adapters"], f'Cluster {cluster_label} - Learning Rate Adapters Distribution', f'cluster_{cluster_label}_lra_histogram.png')
def main():
    USER_ENV_VAR = os.getenv('USER', 'psomu3')
    parser = argparse.ArgumentParser(description='t-SNE Visualization of Genome Embeddings')
    parser.add_argument('--input', type=str, 
                       default='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset/mix_dataset_reg_train.pkl',
                       help='Path to input pickle file')
    parser.add_argument('--output_dir', type=str,
                       default=f'/storage/ice-shared/vip-vvk/data/AOT/{USER_ENV_VAR}/codestral/tsne_visualizations',
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
    parser.add_argument('--classification', type=bool, default=False,
                       help='Whether we run a classification algorithm on tsne models. Default is False.')
    parser.add_argument('--inputHash', type=str, 
                       default='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset/full_out.csv',
                       help='Path to strings of genomes')
    parser.add_argument('--surrogate_weights', type=str, 
                       default='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/surrogate_weights_codestral/surrogate_weights',
                       help='Path to surrogate model weights')

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

    """Test surrogate model on genome matrix and save predictions"""
    os.makedirs(args.output_dir, exist_ok=True)
    testing_dir = f"psomu3/codestral/surrogate_training"
    repo_dir = "/storage/ice-shared/vip-vvk/data/AOT/"
    surrogate = Surrogate('conf.toml', '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/surrogate_weights_codestral/surrogate_weights')
    #predict surrogate values from df["genome"] and append that to df
    genomes = df['genome'].tolist()
    #convert genomes to torch tensors
    genome_tensors = []
    for genome in genomes:
        genome_tensors.append(torch.tensor(genome, dtype=torch.float32))
    genome_batch = torch.stack(genome_tensors)
    with torch.no_grad():
        predictions = surrogate.predict(genome_batch).cpu().numpy()
    #add predictions to df
    df['surrogate_prediction_ciou_loss'] = predictions[:, 0]
    df['surrogate_prediction_average_precision'] = predictions[:, 1]
    df['surrogate_ciou_loss_error'] = abs(df['ciou_loss'] - df['surrogate_prediction_ciou_loss'])
    df['surrogate_average_precision_error'] = abs(df['average_precision'] - df['surrogate_prediction_average_precision'])
    print(df.head())

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
    if args.classification:
        cluster_to_hashes = classify_tsne(tsne_embedding, args.output_dir, args.output_prefix, args.inputHash)
        print(f"Cluster to Hashes mapping saved to 'tsne_cluster_to_hashes_mapping.csv'")
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
