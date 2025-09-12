#!/usr/bin/env python3
"""
Script to generate final .safetensors file from trained LoRA weights
This script processes the trained LoRA weights and saves them in the final format.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_lora_weights(checkpoint_path: str) -> dict:
    """Load LoRA weights from checkpoint."""
    logger.info(f"Loading LoRA weights from {checkpoint_path}")
    
    if checkpoint_path.endswith('.safetensors'):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    return state_dict

def process_lora_weights(state_dict: dict, rank: int = 16) -> dict:
    """Process LoRA weights for final format."""
    processed_dict = {}
    
    for key, value in state_dict.items():
        # Ensure the tensor is on CPU
        if isinstance(value, torch.Tensor):
            value = value.cpu()
        
        # Add metadata to the key
        processed_key = f"lora_{key}"
        processed_dict[processed_key] = value
    
    # Add metadata
    metadata = {
        "training_info": "LoRA fine-tuned with Flux model",
        "rank": str(rank),
        "framework": "pytorch",
        "training_type": "lora",
    }
    
    return processed_dict, metadata

def save_final_safetensors(state_dict: dict, metadata: dict, output_path: str):
    """Save the final .safetensors file."""
    logger.info(f"Saving final .safetensors file to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save with metadata
    save_file(state_dict, output_path, metadata)
    
    logger.info(f"Successfully saved .safetensors file to {output_path}")

def validate_safetensors(file_path: str):
    """Validate the generated .safetensors file."""
    logger.info(f"Validating .safetensors file: {file_path}")
    
    try:
        # Try to load the file
        state_dict = load_file(file_path)
        
        # Check if it contains LoRA weights
        lora_keys = [k for k in state_dict.keys() if k.startswith("lora_")]
        
        if not lora_keys:
            logger.warning("No LoRA weights found in the file")
            return False
        
        logger.info(f"Validation successful. Found {len(lora_keys)} LoRA weight tensors")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate final .safetensors file from trained LoRA weights")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the trained LoRA weights checkpoint")
    parser.add_argument("--output_path", type=str, default="output/lora_weights_final.safetensors",
                       help="Path to save the final .safetensors file")
    parser.add_argument("--rank", type=int, default=16,
                       help="LoRA rank used during training")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the generated .safetensors file")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # Load LoRA weights
    state_dict = load_lora_weights(args.checkpoint_path)
    
    # Process weights
    processed_dict, metadata = process_lora_weights(state_dict, args.rank)
    
    # Save final .safetensors file
    save_final_safetensors(processed_dict, metadata, args.output_path)
    
    # Validate if requested
    if args.validate:
        if validate_safetensors(args.output_path):
            logger.info("Generated .safetensors file is valid!")
        else:
            logger.error("Generated .safetensors file validation failed!")
            sys.exit(1)
    
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()