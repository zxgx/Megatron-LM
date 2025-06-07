import re
import csv
import argparse
import os


# Regular expressions to match the required data
memory_pattern = re.compile(
    r'\[Rank (\d+)\].*?\(after \d+ iterations\) memory \(MB\) \|'
    r' allocated: ([\d.]+) \|'
    r' max allocated: ([\d.]+) \|'
    r' reserved: ([\d.]+) \|'
    r' max reserved: ([\d.]+)'
)

flops_pattern = re.compile(r'iteration\s+(\d+)/\s+\d+.*?elapsed time per iteration \(ms\): ([\d.]+).*?model tflops: ([\d.]+), hardware tflops: ([\d.]+)')
def main(log_dir, pp_size, output_dir):
    file_names = [f"{log_dir}-{rank}-{pp_size}.log" for rank in range(pp_size)]
    
    dir_name, file_name = os.path.split(log_dir)
    if output_dir is not None:
        dir_name = output_dir
        os.makedirs(dir_name, exist_ok=True)
    
    memory_csv_path = os.path.join(dir_name, file_name+'-memory.csv')
    flops_csv_path = os.path.join(dir_name, file_name+'-flops.csv')

    # Lists to store extracted data
    memory_data = []
    flops_data = []

    for file in file_names:
        memory_log, flops_log = parse_log(file)
        memory_data.extend(memory_log)
        flops_data.extend(flops_log)

    # Sort memory data by rank
    memory_data.sort(key=lambda x: x[0])
    if len(memory_data) == 0:
        assert len(flops_data)==0
        print(f"WARNING: file {file_name} is empty.")
    elif len(memory_data) != pp_size*8:
        print(f"WARNING: MEMORY records for file {memory_csv_path} only has {len(memory_data)} entries.")
    if len(flops_data) != 0 and len(flops_data) != 3:
        print(f"WARNING: FLOPS records for file {flops_csv_path} only has {len(flops_data)} entries.")
    
    # Write the memory data to memory.csv
    with open(memory_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write headers for memory data
        csvwriter.writerow(['Rank', 'Allocated', 'Max Allocated', 'Reserved', 'Max Reserved'])
        csvwriter.writerows(memory_data)

    # Write the flops data to flops.csv
    with open(flops_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write headers for flops data
        csvwriter.writerow(['Iteration', 'Elapsed Time (s)', 'Model TFLOPS', 'Hardware TFLOPS'])
        csvwriter.writerows(flops_data)

    # print(f'Memory data has been written to {memory_csv_path}')
    # print(f'Flops data has been written to {flops_csv_path}')

def parse_log(log_file_path):
    # Lists to store extracted data
    memory_log = {} # dedup for adapipe log file
    flops_log = []

    # Read the log file and extract data
    with open(log_file_path, 'r') as file:
        for line in file:
            memory_match = memory_pattern.finditer(line)
            for match in memory_match:
                # Convert rank to int for proper sorting
                rank = int(match.group(1))
                allocated = float(match.group(2))
                max_allocated = float(match.group(3))
                reserved = float(match.group(4))
                max_reserved = float(match.group(5))
                memory_log[rank] = [rank, allocated, max_allocated, reserved, max_reserved]
            
            flops_match = flops_pattern.search(line)
            if flops_match:
                iteration = int(flops_match.group(1))
                elapsed_time = float(flops_match.group(2)) / 1000
                model_tflops = float(flops_match.group(3))
                hardware_tflops = float(flops_match.group(4))
                flops_log.append([iteration, elapsed_time, model_tflops, hardware_tflops])

    return list(memory_log.values()), flops_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract memory and flops data from a log file')
    parser.add_argument('--log-dir', type=str, help='Path to the log file')
    parser.add_argument("--pp-size", type=int)
    parser.add_argument('--output-dir', type=str, help='Path to the output directory', default=None)
    
    args = parser.parse_args()
    main(args.log_dir, args.pp_size, args.output_dir)
