import json  

def get_config(config_path="config.json"):
    """Read the configuration file in JSON format
"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Configuration file loaded successfully！")
        print(json.dumps(config, indent=2))  
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file {config_path} does not exist, please check the path.")
    except json.JSONDecodeError:
        raise ValueError(f"The configuration file {config_path} has an invalid format, please check the JSON syntax.")


