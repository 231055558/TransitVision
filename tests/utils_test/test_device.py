import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.utils import DeviceConfig
import torch

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml"

def test_device():
    print("=== DeviceConfig Test ===")
    
    cfg = DeviceConfig(CONFIG_PATH)
    
    print(f"Device type: {cfg.device_type}")
    print(f"Device id: {cfg.device_id}")
    print(f"Num threads: {cfg.num_threads}")
    print(f"Device: {cfg.device}")
    print(f"Device str: {cfg.device_str}")
    
    tensor = torch.randn(10, 10)
    print(f"\nTensor device before: {tensor.device}")
    
    tensor = tensor.to(cfg.device)
    print(f"Tensor device after: {tensor.device}")
    
    print("\n✓ DeviceConfig works")

if __name__ == "__main__":
    test_device()

