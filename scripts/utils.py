#!/usr/bin/env python3
"""
Utility functions for LoRA fine-tuning
"""

import os
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check if GPU is available and return info."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        logger.info(f"GPU detected: {device_name}")
        logger.info(f"GPU memory: {device_memory:.1f} GB")
        return True, device_name, device_memory
    else:
        logger.warning("No GPU detected. Training will be very slow on CPU.")
        return False, None, 0

def setup_logging(log_dir: str = "logs"):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )

def validate_dataset(image_dir: str, captions_file: str) -> bool:
    """Validate that all images in captions file exist."""
    if not os.path.exists(captions_file):
        logger.error(f"Captions file not found: {captions_file}")
        return False
    
    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        return False
    
    # Read captions file
    with open(captions_file, 'r') as f:
        lines = f.readlines()
    
    missing_images = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                image_name = parts[0].strip()
                image_path = os.path.join(image_dir, image_name)
                
                if not os.path.exists(image_path):
                    missing_images.append(image_name)
    
    if missing_images:
        logger.error(f"Missing images: {missing_images}")
        return False
    
    logger.info("Dataset validation passed!")
    return True

def get_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, cached
    return 0, 0

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def estimate_training_time(num_images: int, epochs: int, batch_size: int, 
                          steps_per_image: int = 1) -> float:
    """
    Estimate training time in hours.
    This is a rough estimate based on typical training speeds.
    """
    # Rough estimate: 1 second per step on RTX 1080
    total_steps = (num_images / batch_size) * epochs * steps_per_image
    estimated_seconds = total_steps * 1.0  # Conservative estimate
    estimated_hours = estimated_seconds / 3600
    
    return estimated_hours

def print_system_info():
    """Print system information for debugging."""
    import platform
    import psutil
    
    logger.info("=== System Information ===")
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")
    
    # CPU info
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"CPU cores: {cpu_count}")
    logger.info(f"RAM: {memory_gb:.1f} GB")
    
    # GPU info
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {device_name} ({device_memory:.1f} GB)")
    else:
        logger.info("No GPU available")

def create_sample_prompts(num_prompts: int = 5) -> List[str]:
    """Create sample prompts for testing the trained LoRA."""
    base_prompts = [
        "a photo of [your_subject]",
        "[your_subject] in a garden",
        "[your_subject] at sunset",
        "a close-up photo of [your_subject]",
        "[your_subject] with a beautiful background",
        "artistic portrait of [your_subject]",
        "[your_subject] in natural lighting",
        "a professional photo of [your_subject]",
        "[your_subject] looking at the camera",
        "[your_subject] in a studio setting"
    ]
    
    return base_prompts[:num_prompts]

def save_training_config(config: Dict[str, Any], output_dir: str):
    """Save training configuration for reproducibility."""
    import yaml
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as YAML
    yaml_path = os.path.join(output_dir, "training_config.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save as JSON
    json_path = os.path.join(output_dir, "training_config.json")
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training configuration saved to {output_dir}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'torch', 'torchvision', 'transformers', 'diffusers', 
        'accelerate', 'safetensors', 'PIL', 'numpy', 'tqdm',
        'peft', 'omegaconf', 'hydra'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install them using: pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are installed!")
    return True