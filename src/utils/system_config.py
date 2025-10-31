"""
System configuration utilities.
"""

import os
import multiprocessing


def configure_system():
    """
    Configure system-level settings for optimal performance.
    Sets thread counts for scientific libraries and TensorFlow.
    """
    try:
        num_threads = multiprocessing.cpu_count()
    except NotImplementedError:
        num_threads = 4
    
    # Configure scientific libraries
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    
    print(f"Scientific libraries configured to use {num_threads} threads.")

