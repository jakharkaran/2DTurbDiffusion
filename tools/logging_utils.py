"""
Logging utilities for distributed training/sampling
"""
import os

def get_rank():
    """Get the current process rank."""
    try:
        return int(os.environ.get("LOCAL_RANK", 0))
    except:
        return 0

def log_print(message, log_to_screen=True, print_all_devices=0, end='\n'):
    """
    Print function that respects distributed training and config flags.
    
    Args:
        message: The message to print
        log_to_screen: Whether log_to_screen is enabled (from config)
        force: If True, print regardless of log_to_screen setting (for critical messages)
        end: Line ending character (default: newline)
    """

    rank = get_rank()
    # If print_all_devices is 'all', print on all devices
    if print_all_devices == 'all' and log_to_screen:
        print(f"[Device ID {rank}] {message}", end=end)
    else:
        # Get the rank of the current process
        if rank == 0 and log_to_screen:
            print(f"[Device ID {rank}] {message}", end=end)