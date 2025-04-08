import sys, subprocess
import yaml
from ruamel.yaml import YAML

def main(model_collapse_type, model_collapse_gen, config_path='config/config'):
    """
    Load the YAML configuration preserving formatting and comments,
    update train_params values, and write the updated configuration to a new file.
    """
    yaml = YAML(typ='rt') # ruamel.yaml.YAML() instance
    # Load the current configuration
    with open(config_path + '.yaml', 'r') as f:
        config = yaml.load(f)
    
    # Update the specific values in the configuration
    config['train_params']['model_collapse_type'] = model_collapse_type
    config['train_params']['model_collapse_gen'] = model_collapse_gen
    
    # Determine the target filename for saving the updated configuration
    target_file = f"{config_path}_{model_collapse_type}_{model_collapse_gen}.yaml"
    
    # Write the updated configuration to the target file; ruamel.yaml preserves comments and formatting by default
    with open(target_file, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Updated configuration saved to {target_file}")

if __name__ == '__main__':
    model_collapse_type = str(sys.argv[1])
    model_collapse_gen = int(sys.argv[2])
    config_path = 'config/config'
    main(model_collapse_type, model_collapse_gen, config_path)
