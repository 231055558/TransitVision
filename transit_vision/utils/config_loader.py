import yaml
from pathlib import Path

def load_config(config_path_str: str):
    # 加载并解析指定的 YAML 配置文件
    config_path = Path(config_path_str)
    
    if not config_path.is_file():
        print(f"Error: Configuration file not found at '{config_path}'")
        if not config_path.is_absolute():
            root_path = Path(__file__).resolve().parent.parent.parent
            alt_path = root_path / config_path_str
            if alt_path.is_file():
                print(f"Found config file at project root: '{alt_path}'")
                config_path = alt_path
            else:
                return None
        else:
            return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from '{config_path}'")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{config_path}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading config '{config_path}': {e}")
        return None
