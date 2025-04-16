import os
import re
import glob

def concatenate_lambda_search_files(directory_path, output_file):
    """
    Find all files matching the pattern 'scores_lambda_search_[integer]_[integer].txt'
    in the given directory, read their contents, and concatenate them into one file.
    
    Args:
        directory_path (str): Path to the directory containing the files
        output_file (str): Path where the combined output will be saved
    
    Returns:
        int: Number of files processed
    """
    # Ensure the directory path exists
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory path does not exist: {directory_path}")
    
    # Pattern to match: scores_lambda_search_[integer]_[integer].txt
    pattern = os.path.join(directory_path, "scores_lambda_search_*_*.txt")
    
    # Find all matching files
    matching_files = glob.glob(pattern)
    
    # Filter to ensure we only get files matching the exact pattern with integers
    file_pattern = re.compile(r'scores_lambda_search_\d+_\d+\.txt$')
    matching_files = [f for f in matching_files if file_pattern.search(os.path.basename(f))]
    
    # Sort files to ensure consistent ordering
    matching_files.sort()
    
    if not matching_files:
        print(f"No matching files found in {directory_path}")
        return 0
    
    # Concatenate all files into one
    with open(output_file, 'w') as outfile:
        for file_path in matching_files:
            print(f"Processing: {file_path}")
            with open(file_path, 'r') as infile:
                outfile.write(infile.read())
                # Add a newline between files to ensure clean separation
                outfile.write('\n')
    
    print(f"Successfully concatenated {len(matching_files)} files into {output_file}")
    return len(matching_files)

# Usage example
ROOT_DIR = "/storage/ice-shared/vip-vvk/data/AOT"
source_dir = os.path.join(ROOT_DIR, "psomu3/uda/grad_regu")
output_file = os.path.join(source_dir, "all_lambda_search_results.txt")

num_files = concatenate_lambda_search_files(source_dir, output_file)
print(f"Total files processed: {num_files}")

def read_file_sections(filename):
    """Read a file and split it into sections by lambda value."""
    sections = {}
    
    with open(filename, 'r') as file:
        content = file.read()
    
    # Split by lambda headers
    lambda_sections = re.split(r'(Reg lambda is [^{]+)', content)
    
    # Skip first empty section if exists
    if lambda_sections[0].strip() == '':
        lambda_sections = lambda_sections[1:]
    
    # Process pairs (header + content)
    for i in range(0, len(lambda_sections), 2):
        if i+1 < len(lambda_sections):
            header = lambda_sections[i].strip()
            content = lambda_sections[i+1].strip()
            sections[header] = content
    
    return sections

def merge_lambda_results(file1, file2, output_file):
    """Merge lambda results from two files into one."""
    sections1 = read_file_sections(file1)
    sections2 = read_file_sections(file2)
    
    with open(output_file, 'w') as outfile:
        # Get all unique lambda headers
        all_headers = sorted(set(list(sections1.keys()) + list(sections2.keys())), key=lambda x: float(x.split()[-1]))
        
        for header in all_headers:
            outfile.write(header + "\n")
            
            # Add content from file1 if this lambda exists there
            if header in sections1:
                outfile.write(sections1[header] + "\n")
            
            # Add content from file2 if this lambda exists there
            if header in sections2:
                outfile.write(sections2[header] + "\n")
            

second_file = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/uda/grad_regu/scores_lambda_search_first_10.txt"
final_output_file = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/uda/grad_regu/merged_lambda_results.txt"

merge_lambda_results(output_file, second_file, final_output_file)