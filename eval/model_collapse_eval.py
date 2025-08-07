import yaml
import subprocess
import os

model_collapse_type = 'all_gen'

# Load config file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Loop through numbers 1 to 5
for i in range(1, 7):
    # Update root_dir with current number
    config['root_dir'] = f"/home/exouser/karan2/2DTurbDiffusion/results/model_collapse/{model_collapse_type}/UDM_s3_Oaw_lrC3e4_{model_collapse_type}_{i}"
    
    # Save updated config
    with open('config/config.yaml', 'w') as file:
        yaml.dump(config, file)
    
    print(f"Running analyze.py with root_dir ending in {i}")
    
    # Run analyze.py
    result = subprocess.run(['python', 'analyze_model.py'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Successfully completed run {i}")
    else:
        print(f"Error in run {i}: {result.stderr}")
