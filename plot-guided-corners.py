import json
import argparse
import matplotlib.pyplot as plt

def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.
    
    :param file_path: Path to the JSONL file.
    :return: List of dictionaries containing data from each JSON line.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def extract_points(data):
    """
    Extracts 2D points from the data.
    
    :param data: List of dictionaries containing 'points2d'.
    :return: List of tuples representing 2D points.
    """
    points = []
    for entry in data:
        for point in entry['points2d']:
            points.append(point['pixel'])
    return points

def plot_points(points, width, height):
    """
    Plots the 2D points using matplotlib with specified bounds.
    
    :param points: List of tuples representing 2D points.
    :param width: Width bound for the plot.
    :param height: Height bound for the plot.
    """
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    
    plt.scatter(x_values, y_values, c='blue', marker='o', alpha=0.3, s=1)
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of 2D Points')
    plt.grid(True)
    plt.show()

def main(file_path, resolution):
    # Parse the resolution argument
    width, height = map(int, resolution.split('x'))
    
    data = read_jsonl(file_path)
    points = extract_points(data)
    plot_points(points, width, height)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2D points from a JSONL file with specified resolution.")
    parser.add_argument('file_path', type=str, help="Path to the JSONL file.")
    parser.add_argument('--resolution', type=str, required=True, help="Resolution for the scatter plot in format WIDTHxHEIGHT.")
    args = parser.parse_args()
    main(args.file_path, args.resolution)
