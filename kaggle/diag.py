import os, subprocess
print("nvidia-smi:", flush=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout, flush=True)
import torch
print("torch:", torch.__version__, flush=True)
print("arch_list:", torch.cuda.get_arch_list(), flush=True)
print("device_capability:", torch.cuda.get_device_capability(), flush=True)
print("device_name:", torch.cuda.get_device_name(0), flush=True)
try:
    x = torch.randn(4, 4, device="cuda")
    y = (x @ x).sum().item()
    print("matmul OK:", y, flush=True)
except Exception as e:
    print("matmul FAILED:", repr(e), flush=True)
