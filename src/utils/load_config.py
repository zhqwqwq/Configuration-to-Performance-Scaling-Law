import argparse
import yaml
from omegaconf import OmegaConf

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='configs/train_config.yaml', help='Path to YAML config file')
    parser.add_argument('--override', nargs='*', default=[], help='List of key=value pairs to override YAML config')
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as f:
        base_cfg = OmegaConf.create(yaml.safe_load(f))
    cli_cfg = OmegaConf.from_dotlist(args.override)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    return cfg
