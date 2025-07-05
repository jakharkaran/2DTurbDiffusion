import os
import torch
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    # Set the device first, then initialize process group
    local_rank = int(os.environ["LOCAL_RANK"]) # Environment variable provided by torchrun
    torch.cuda.set_device(local_rank)
    
    # Verify device is set correctly
    current_device = torch.cuda.current_device()
    if current_device != local_rank:
        raise RuntimeError(f"Device mismatch: expected {local_rank}, got {current_device}")
    
    # Initialize process group - the device should be automatically detected
    # from the current CUDA device set above
    init_process_group(backend="nccl")

def ddp_cleanup():
    if torch.distributed.is_initialized():
        destroy_process_group()