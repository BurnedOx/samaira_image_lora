#!/usr/bin/env python3
"""
Main entry point for LoRA fine-tuning with Flux model
This script provides a simple interface to train LoRA with 20 images.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from scripts.utils import (
    check_gpu_availability,
    setup_logging,
    validate_dataset,
    print_system_info,
    check_dependencies,
    estimate_training_time,
    create_sample_prompts
)

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train LoRA with Flux model using 20 images")
    parser.add_argument("--data_dir", type=str, default="data/images",
                       help="Directory containing the 20 training images")
    parser.add_argument("--captions_file", type=str, default="data/captions.txt",
                       help="Path to captions file")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory for trained model")
    parser.add_argument("--config_dir", type=str, default="configs",
                       help="Directory containing configuration files")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--validate_only", action="store_true",
                       help="Only validate the dataset and exit")
    parser.add_argument("--system_info", action="store_true",
                       help="Print system information and exit")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be done without actually training")
    
    args = parser.parse_args()
    
    # Print system info if requested
    if args.system_info:
        print_system_info()
        return
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Please install them first.")
        sys.exit(1)
    
    # Check GPU availability
    gpu_available, gpu_name, gpu_memory = check_gpu_availability()
    
    if not gpu_available:
        logger.error("GPU is required for training. Please check your CUDA installation.")
        sys.exit(1)
    
    if gpu_memory < 8:
        logger.warning(f"GPU memory is low ({gpu_memory:.1f} GB). Training might be slow.")
    
    # Validate dataset
    logger.info("Validating dataset...")
    if not validate_dataset(args.data_dir, args.captions_file):
        logger.error("Dataset validation failed. Please check your data.")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Dataset validation passed. Exiting.")
        return
    
    # Estimate training time
    num_images = 20  # Fixed to 20 images as per requirement
    estimated_time = estimate_training_time(
        num_images=num_images,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    logger.info(f"Estimated training time: {estimated_time:.1f} hours")
    
    if args.dry_run:
        logger.info("Dry run completed. No training was performed.")
        logger.info(f"Would train for {args.epochs} epochs with batch size {args.batch_size}")
        logger.info(f"Output would be saved to: {args.output_dir}")
        return
    
    # Import training script here to avoid import errors if dependencies are missing
    try:
        from scripts.train_lora import main as train_main
    except ImportError as e:
        logger.error(f"Failed to import training script: {e}")
        sys.exit(1)
    
    # Build training arguments
    train_args = [
        "--training_config", f"{args.config_dir}/training_config.yaml",
        "--dataset_config", f"{args.config_dir}/dataset_config.yaml",
        "--captions_file", args.captions_file,
        "--output_dir", args.output_dir,
    ]
    
    # Update sys.argv for the training script
    original_argv = sys.argv
    sys.argv = ["train_lora.py"] + train_args
    
    try:
        logger.info("Starting LoRA training...")
        train_main()
        logger.info("Training completed successfully!")
        
        # Generate final .safetensors file
        logger.info("Generating final .safetensors file...")
        from scripts.generate_safetensors import main as generate_main
        
        generate_args = [
            "--checkpoint_path", f"{args.output_dir}/lora_weights.safetensors",
            "--output_path", f"{args.output_dir}/lora_weights_final.safetensors",
            "--rank", str(args.rank),
            "--validate",
        ]
        
        sys.argv = ["generate_safetensors.py"] + generate_args
        generate_main()
        
        logger.info("Final .safetensors file generated successfully!")
        logger.info(f"Your trained LoRA is available at: {args.output_dir}/lora_weights_final.safetensors")
        
        # Print sample prompts for testing
        logger.info("\n=== Sample prompts for testing your LoRA ===")
        sample_prompts = create_sample_prompts()
        for i, prompt in enumerate(sample_prompts, 1):
            logger.info(f"{i}. {prompt}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()