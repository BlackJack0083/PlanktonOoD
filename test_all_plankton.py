import subprocess
import os
import glob
import pandas as pd
import argparse
import sys
import logging
import io 

# ====== Setup Logging ======
# The Tee class only captures print() output from the parent Python script.
# Subprocess output needs to be captured in real-time and logged separately via run_and_log.
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    filename='output.log',
    filemode='w'
)

# ====== Redirect print() to Terminal + Log (for logging main script print calls) ======
class Tee:
    """
    Tee class redirects sys.stdout and sys.stderr of the main script, simultaneously 
    outputting to the terminal and the log file. Subprocess output is handled 
    separately in run_and_log.
    """
    def __init__(self, stream, logger_func):
        self.stream = stream
        self.logger_func = logger_func

    def write(self, message):
        # Log to file if it is not an empty string or only contains newlines/carriage returns
        log_message = message.strip()
        if log_message:
            self.logger_func(log_message)
        
        # Always write to the original stream (terminal) for display assurance
        self.stream.write(message)

    def flush(self):
        self.stream.flush()

# Redirect the main script's print() output to Tee
sys.stdout = Tee(sys.__stdout__, logging.info)
sys.stderr = Tee(sys.__stderr__, logging.error)


def run_and_log(command):
    """
    Runs a subprocess command, captures its output in real-time, logs it to a file, 
    and preserves dynamic progress bar effects.
    """
    display_command = command.replace('PYTHONIOENCODING=utf-8 stdbuf -oL ', '')
    print(f"\n====== Running Command: {display_command} ======")
    
    # Start subprocess using Popen, setting up pipes to capture output
    # stderr=subprocess.STDOUT merges standard error into standard output for unified handling
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
        )
    except FileNotFoundError:
        logging.error(f"Command execution failed: executable or script not found. Command: {command}")
        print(f"\n--- ERROR: Command execution failed (file or script not found) ---", file=sys.stderr)
        return
    
    # Variable for buffering log content
    log_buffer = ""

    # Use io.TextIOWrapper to decode byte stream in real-time
    # newline='' prevents TextIOWrapper from automatically handling newlines, ensuring \r is preserved
    with io.TextIOWrapper(process.stdout, encoding='utf-8', errors='replace', newline='') as stdout_reader:
        
        while True:
            # Read in small chunks (e.g., 100 chars) instead of waiting for a full line, which is key to preserving progress bars.
            chunk = stdout_reader.read(100) 
            
            # Exit loop if no more output and process has terminated
            if not chunk and process.poll() is not None:
                break
            
            if chunk:
                # 1. Real-time output to terminal
                # Write directly to original stdout, preserving all dynamic characters (\r)
                sys.__stdout__.write(chunk)
                sys.__stdout__.flush()

                # 2. Logging Logic: Buffer data until a complete line (\n) is found
                log_buffer += chunk
                
                # Process all complete lines
                while '\n' in log_buffer:
                    # Find the first newline
                    line, log_buffer = log_buffer.split('\n', 1)
                    
                    # Log the complete line. 
                    # Remove \r (carriage return), as it is a terminal control character and should not appear in the log.
                    final_line = line.rstrip('\r') 
                    
                    # Only log if the line contains non-whitespace content
                    if final_line.strip():
                        logging.info(final_line)

    # 3. Log any remaining content (the final line, possibly without \n)
    if log_buffer.strip():
        # Remove \r, and log the final non-empty line
        final_line = log_buffer.rstrip('\r')
        if final_line.strip():
            logging.info(final_line)

    # Ensure the subprocess fully terminates
    process.wait()
    
    if process.returncode != 0:
        logging.error(f"Command failed with exit code: {process.returncode}. Command: {command}")
        # Use raw sys.__stderr__.write() to ensure the error message bypasses the Tee class and is displayed on the terminal
        sys.__stderr__.write(f"\n--- WARNING: Command failed (exit code: {process.returncode}). Please check the log file --- \n")
        sys.__stderr__.flush()


# -------------------------------
# 1. Define aliases for all methods
# -------------------------------
aliases = [
    "msp",      # post-hoc
    "vim",
    "ash",      # post-hoc
    # "cider",    # training
    # "conf_branch",  # training
    "ebo",      # post-hoc, energy
    "odin",     # post-hoc
    "mds",      # post-hoc
    # "mds_ensemble", # post-hoc
    # "npos",     # training 
    "rmds",     # post-hoc
    #"gmm",     # post-hoc
    #"patchcore",    # post-hoc
    "openmax",  # post-hoc
    "react",    # post-hoc
    "gradnorm", # post-hoc
    # "godin",    # training
    # "gram",     # post-hoc
    # "cutpaste", # ?
    "mls",      # post-hoc
    "residual", # Deep Residual Flow for Out of Distribution Detection
    "klm",      # post-hoc
    "temperature_scaling",  # post-hoc
    # "ensemble", 
    "dropout",  # post-hoc
    # "draem",    # ?
    # "dsvdd",    # ?
    # "mos",      # training
    # "mcd",      # training
    # "opengan",  # post-hoc
    "knn",      # post-hoc
    "dice",     # post-hoc
    # "ssd",      # SSD: A Unified Framework for Self-Supervised Outlier Detection
    "she",      # post-hoc
    # "rd4ad",    # posthoc
    # "rotpred",  # training
    "rankfeat", # post-hoc
    "temp_scaling", # post-hoc
    # Complete missing items (comments can be added as needed)
    "fdbd",     # post-hoc
    # "rts",      # post-hoc
    "gen",      # post-hoc
    "relation", # post-hoc
    # "t2fnorm",  # post-hoc
]


# -------------------------------
# 2. Evaluation Parameter Setup
# -------------------------------
id_data = "plankton54"
# Base results directory: to prevent output files from different methods from overwriting each other, 
# each method will output to the base_root/<alias> subdirectory
base_root = "results/plankton54_resnet50_base_e100_lr0.1_default"
# Common arguments
common_args = f"--id-data {id_data} --save-score --save-csv"

# -------------------------------
# 3. Iterate through all aliases for evaluation
# -------------------------------
for alias in aliases:
    result_dir = base_root
    
    # Add PYTHONIOENCODING=utf-8 to ensure the subprocess outputs correct Unicode characters.
    command = f'PYTHONIOENCODING=utf-8 stdbuf -oL python scripts/eval_ood.py {common_args} --root {result_dir} --postprocessor "{alias}"'
    
    # Use the modified run_and_log function
    run_and_log(command)

# -------------------------------
# 4. Merge all evaluation results (CSV files)
# -------------------------------
print("\n====== Merging Evaluation Results ======")
csv_pattern = os.path.join(base_root, "ood", "*.csv")
csv_files = glob.glob(csv_pattern)
dfs = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        # Extract filename (without extension) as method alias, and add an identifier column
        method_name = os.path.splitext(os.path.basename(csv_file))[0]
        df["method"] = method_name
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_csv_path = os.path.join(base_root, "merged_results.csv")
    merged_df.to_csv(merged_csv_path, index=False)
    print("Evaluation results for all methods have been merged and saved to:", merged_csv_path)
else:
    print("No CSV files found. Please check the evaluation results for each method.")
