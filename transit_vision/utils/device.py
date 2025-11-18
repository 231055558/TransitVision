import yaml
import torch
from pathlib import Path

class DeviceConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        self.device_type = cfg['device_type']
        self.device_id = cfg['device_id']
        self.num_threads = cfg['num_threads']
        self.batch_size = cfg.get('batch_size', 1)
        self.fp16 = cfg.get('fp16', True)
        
        self._init_device()
    
    def _init_device(self):
        if self.device_type == 'npu':
            import torch_npu
            from torch_npu.contrib import transfer_to_npu
            self.device = torch.device(f'npu:{self.device_id}')
        else:
            self.device = torch.device(f'cuda:{self.device_id}')
        
        torch.set_num_threads(self.num_threads)
    
    def to_device(self, model):
        return model.to(self.device)
    
    @property
    def device_str(self):
        return f'{self.device_type}:{self.device_id}'

