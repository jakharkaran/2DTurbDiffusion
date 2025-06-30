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

def log_print(message, log_to_screen=True, end='\n'):
    """
    Print function that respects distributed training and config flags.
    
    Args:
        message: The message to print
        log_to_screen: Whether log_to_screen is enabled (from config)
        force: If True, print regardless of log_to_screen setting (for critical messages)
        end: Line ending character (default: newline)
    """
    rank = get_rank()
    
    # Only rank 0 prints, and only if verbose is enabled
    if rank == 0 and log_to_screen:
        print(message, end=end)
