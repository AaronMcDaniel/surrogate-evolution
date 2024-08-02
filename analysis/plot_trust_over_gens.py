import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.ndimage import uniform_filter1d

# Function to compute moving average
def moving_average(data, window_size):
    return uniform_filter1d(data, size=window_size)

def main(input_file, output_file, window_size):
    # Load the list of tuples from the pickle file
    with open(input_file, 'rb') as file:
        trust_data = pickle.load(file)

    # Extract classification and regressor trusts
    generations = list(range(1, len(trust_data) + 1))
    cls_trusts = [t[0] for t in trust_data]
    reg_trusts = [t[1] for t in trust_data]

    # Calculate moving averages
    cls_trusts_smooth = moving_average(cls_trusts, window_size)
    reg_trusts_smooth = moving_average(reg_trusts, window_size)

    # Plot the data
    plt.figure(figsize=(10, 6))

    # Classification trust plot
    plt.plot(generations, cls_trusts, marker='o', linestyle='-', color='b', alpha=0.3, label='Classification Trust')
    plt.plot(generations, cls_trusts_smooth, color='b', linewidth=2, label='Smoothed Classification Trust')

    # Regressor trust plot
    plt.plot(generations, reg_trusts, marker='o', linestyle='-', color='r', alpha=0.3, label='Regressor Trust')
    plt.plot(generations, reg_trusts_smooth, color='r', linewidth=2, label='Smoothed Regressor Trust')

    plt.xlabel('Generation')
    plt.ylabel('Trust')
    plt.title('Trust Over Generations')
    plt.grid(True)
    plt.legend()

    # Set x-axis ticks to show every generation
    plt.xticks(generations)

    # Save the plot to the output file
    plt.savefig(output_file)
    plt.close()

    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Trust Over Generations')
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Path to the input pickle file')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Path to the output image file')
    parser.add_argument('-w', '--window_size', type=int, default=5, help='Window size for moving average')
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_file, args.window_size)