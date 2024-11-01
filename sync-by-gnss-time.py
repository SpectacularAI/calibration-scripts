import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy  # Import the copy module for deepcopy

def read_jsonl(file_path):
    """
    Read JSONL file and return a list of records.
    """
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            obj = json.loads(line)
            if "time" in obj: data.append(obj)
        return data

def write_jsonl(data, file_path):
    """
    Write a list of records to a JSONL file.
    """
    with open(file_path, 'w') as file:
        for record in data:
            json.dump(record, file)
            file.write('\n')

def sync_to_gnss_time(records):
    """
    Synchronize all time fields to GNSS time using linear interpolation.
    """
    # Extract records that have 'gnssTime'
    gnss_records = [r for r in records if 'gnssTime' in r]
    if len(gnss_records) < 2:
        print("Not enough GNSS time points for interpolation.")
        return records

    # Extract times for interpolation
    monotonic_times = np.array([r['time'] for r in gnss_records])
    gnss_times = np.array([r['gnssTime'] for r in gnss_records])

    # Create a function for linear interpolation
    interpolate_fn = np.interp

    # Interpolating GNSS time for each record based on 'time'
    for record in records:
        if 'gnssTime' not in record:
            record_time = record['time']
            gnss_time_interpolated = interpolate_fn(record_time, monotonic_times, gnss_times)
            record['time'] = gnss_time_interpolated  # Sync 'time' to GNSS time

    return records

def plot_times(original_records, adjusted_records):
    """
    Plot original monotonic and GNSS-synced time fields.
    """
    original_times = [r['time'] for r in original_records if 'sensor' in r]
    gnss_synced_times = [r['time'] for r in adjusted_records if 'sensor' in r]

    plt.figure(figsize=(10, 6))
    plt.plot(original_times, label='Original Monotonic Time', marker='o')
    plt.plot(gnss_synced_times, label='GNSS-Synced Time', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Time')
    plt.legend()
    plt.title('Original Monotonic Time vs GNSS-Synced Time')
    plt.show()

def main(input_file, output_file, plot=False):
    # Read records from input file
    records = read_jsonl(input_file)

    # Use deepcopy to store original records for plotting purposes
    original_records = copy.deepcopy(records)

    # Sync times to GNSS time
    adjusted_records = sync_to_gnss_time(records)

    # Write adjusted records to output file
    write_jsonl(adjusted_records, output_file)

    # Plot if requested
    if plot:
        plot_times(original_records, adjusted_records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronize time fields to GNSS time using linear interpolation.")
    parser.add_argument('input_file', help='Input JSONL file to process.')
    parser.add_argument('--out', required=True, help='Output JSONL file to save GNSS-synced times.')
    parser.add_argument('--plot', action='store_true', help='Plot original and GNSS-synced times.')
    args = parser.parse_args()

    main(args.input_file, args.out, args.plot)