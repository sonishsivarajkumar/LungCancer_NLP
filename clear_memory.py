# clear_memory.py

import os
import psutil
import ctypes
import subprocess
import gc
import logging
from datetime import datetime

def setup_logging():
    """Configure logging to track memory cleanup operations"""
    logging.basicConfig(
        filename=f'memory_cleanup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_memory_info():
    """Get current memory usage information"""
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / (1024 ** 3),  # Convert to GB
        'available': memory.available / (1024 ** 3),
        'percent_used': memory.percent
    }

def clear_standby_list():
    """Clear Windows system standby list using RAMMap"""
    try:
        # Note: User needs to download RAMMap from Sysinternals
        rammap_path = r"C:\Path\To\RAMMap.exe"
        if os.path.exists(rammap_path):
            subprocess.run([rammap_path, "-Ew"], check=True)
            return True
    except subprocess.SubprocessError as e:
        logging.error(f"Failed to clear standby list: {e}")
    return False

def clear_python_memory():
    """Clear Python-specific memory"""
    gc.collect()
    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)

def empty_working_set():
    """Empty working set using Windows API"""
    try:
        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        return True
    except Exception as e:
        logging.error(f"Failed to empty working set: {e}")
        return False

def clear_system_memory():
    """Main function to clear system memory"""
    setup_logging()
    
    # Log initial memory state
    initial_memory = get_memory_info()
    logging.info(f"Initial memory state: {initial_memory}")
    
    # Clear Python's internal memory
    clear_python_memory()
    
    # Empty working set
    if empty_working_set():
        logging.info("Successfully emptied working set")
    
    # Clear standby list (requires RAMMap)
    if clear_standby_list():
        logging.info("Successfully cleared standby list")
    
    # Log final memory state
    final_memory = get_memory_info()
    logging.info(f"Final memory state: {final_memory}")
    
    memory_freed = initial_memory['percent_used'] - final_memory['percent_used']
    return {
        'initial_state': initial_memory,
        'final_state': final_memory,
        'memory_freed_percent': memory_freed
    }

if __name__ == "__main__":
    print("Starting memory cleanup process...")
    results = clear_system_memory()
    print(f"\nMemory Cleanup Results:")
    print(f"Initial RAM usage: {results['initial_state']['percent_used']:.1f}%")
    print(f"Final RAM usage: {results['final_state']['percent_used']:.1f}%")
    print(f"Memory freed: {results['memory_freed_percent']:.1f}%")
    print(f"\nAvailable RAM for training: {results['final_state']['available']:.1f} GB")