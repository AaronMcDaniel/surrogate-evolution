"""
Removes all NON-PARETO-OPTIMAL weights from a run directory.
Will only work if you pass in an evolution run directory and it contains a hall_of_fame.csv
"""


import os
import sys
import pandas as pd

def load_hashes(directory):
    # Load the hashes from the hall_of_fame.csv file
    hof_path = os.path.join(directory, 'hall_of_fame.csv')
    if not os.path.isfile(hof_path):
        print(f"Error: {hof_path} does not exist.")
        sys.exit(1)
    
    df = pd.read_csv(hof_path)
    if 'hash' not in df.columns:
        print("Error: 'hash' column not found in hall_of_fame.csv.")
        sys.exit(1)
    
    return set(df['hash'].astype(str))

def delete_pth_files(directory, hashes_to_keep):
    # Traverse the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth'):
                file_path = os.path.join(root, file)
                # Skip deleting if the file is in a directory matching any hash
                parent_dir = os.path.basename(os.path.dirname(file_path))
                if parent_dir in hashes_to_keep:
                    print(f"Skipping file in protected directory: {file_path}")
                    continue
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {str(e)}")

    print('Purge Completed!')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python purge_all_weights.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    # Load hashes to keep from hall_of_fame.csv
    hashes_to_keep = load_hashes(directory)

    # Delete .pth files except in protected directories
    delete_pth_files(directory, hashes_to_keep)