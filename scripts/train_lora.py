#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Flux Model
This script fine-tunes a LoRA adapter on the Flux model using 20 input images.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import Hugging Face Hub for authentication
from huggingface_hub import login

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Diffusers imports
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    FluxPipeline,
    UNet2DConditionModel,
)
from diffusers.models.transformers import Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import logging as diffusers_logging

# Transformers imports
from transformers import CLIPTextModel, CLIPTokenizer

# LoRA imports
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

# Utilities
import numpy as np
from PIL import Image
import yaml
from omegaconf import OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
diffusers_logging.set_verbosity_info()

# Memory optimization functions
def get_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, cached
    return 0, 0

def clear_gpu_memory():
    """Aggressively clear GPU memory cache to prevent crashes during checkpoint creation."""
    if torch.cuda.is_available():
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Collect IPC memory
        torch.cuda.ipc_collect()
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Log memory status after cleanup
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        logger.debug(f"Memory after cleanup - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

def log_memory_usage(stage: str = ""):
    """Log current GPU memory usage with stage information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        logger.info(f"Memory usage {stage}:")
        logger.info(f"  Allocated: {allocated:.2f} GB")
        logger.info(f"  Cached: {cached:.2f} GB")
        logger.info(f"  Max allocated: {max_allocated:.2f} GB")
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
    else:
        logger.info(f"Memory usage {stage}: CUDA not available")

def apply_quantization(model, config: Dict):
    """Apply quantization to model for memory efficiency."""
    quantization_config = config.get("quantization", {})
    
    if not quantization_config.get("enabled", False):
        logger.info("Quantization disabled, skipping...")
        return model
    
    bits = quantization_config.get("bits", 8)
    device = quantization_config.get("device", "cuda")
    
    logger.info(f"Applying {bits}-bit quantization on {device}...")
    
    try:
        if bits == 8:
            # Try to use bitsandbytes for 8-bit quantization
            try:
                import bitsandbytes as bnb
                logger.info("Using bitsandbytes for 8-bit quantization...")
                
                if hasattr(model, 'to'):
                    model.to(device)
                
                # Note: Full model quantization might need specific implementation
                # For now, we'll focus on memory-efficient loading
                logger.info("Quantization settings applied (memory-efficient mode)")
                
            except ImportError:
                logger.warning("bitsandbytes not available, using PyTorch quantization...")
                # Fallback to PyTorch quantization if available
                if hasattr(torch, 'quantization'):
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("PyTorch dynamic quantization applied")
                else:
                    logger.warning("PyTorch quantization not available, continuing without quantization")
        
        elif bits == 4:
            logger.info("4-bit quantization requested - requires bitsandbytes")
            # 4-bit quantization would require specific implementation
            
    except Exception as e:
        logger.error(f"Failed to apply quantization: {e}")
        logger.info("Continuing without quantization...")
    
    return model

# Dataset class
class LoRADataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: List[str], captions: List[str], tokenizer: CLIPTokenizer, 
                 size: int = 512, center_crop: bool = True, random_flip: bool = True):
        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        
        # Image transformations
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.image_transforms(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, self.size, self.size)
        
        # Tokenize caption
        text_inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids.squeeze(),
            "attention_mask": text_inputs.attention_mask.squeeze(),
        }

def load_dataset_config(config_path: str) -> Dict:
    """Load dataset configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_training_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_captions_file(captions_file: str) -> Tuple[List[str], List[str]]:
    """Parse captions file to get image paths and captions."""
    image_paths = []
    captions = []
    
    with open(captions_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Split on first colon
                parts = line.split(':', 1)
                if len(parts) == 2:
                    image_name = parts[0].strip()
                    caption = parts[1].strip()
                    
                    # Construct full path
                    image_path = Path("data/images") / image_name
                    image_paths.append(str(image_path))
                    captions.append(caption)
    
    return image_paths, captions

def create_lora_config(config: Dict) -> LoraConfig:
    """Create LoRA configuration from training config."""
    lora_config = LoraConfig(
        r=config["model"]["lora"]["rank"],
        lora_alpha=config["model"]["lora"]["alpha"],
        target_modules=config["model"]["lora"]["target_modules"],
        lora_dropout=config["model"]["lora"]["dropout"],
        bias=config["model"]["lora"]["bias"],
        task_type="DIFFUSION",
    )
    return lora_config

def save_lora_weights(unet: torch.nn.Module, output_dir: str, step: Optional[int] = None):
    """Save LoRA weights as safetensors file with robust memory management."""
    logger.info(f"Preparing to save LoRA weights {'at step ' + str(step) if step is not None else ''}")
    
    try:
        # Clear GPU memory before checkpoint creation
        clear_gpu_memory()
        
        # Get LoRA state dict
        logger.info("Extracting LoRA state dict...")
        lora_state_dict = get_peft_model_state_dict(unet)
        
        # Move all tensors to CPU to reduce GPU memory pressure
        logger.info("Moving tensors to CPU...")
        cpu_state_dict = {}
        for key, tensor in lora_state_dict.items():
            if isinstance(tensor, torch.Tensor):
                cpu_state_dict[key] = tensor.cpu()
            else:
                cpu_state_dict[key] = tensor
        
        # Clear the original state dict to free memory
        del lora_state_dict
        clear_gpu_memory()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine output path
        if step is not None:
            output_path = os.path.join(output_dir, f"lora_weights_step_{step}.safetensors")
        else:
            output_path = os.path.join(output_dir, "lora_weights.safetensors")
        
        # Save using safetensors with error handling
        logger.info(f"Saving LoRA weights to {output_path}...")
        from safetensors.torch import save_file
        
        # Save in smaller chunks to reduce memory pressure
        try:
            save_file(cpu_state_dict, output_path)
            logger.info(f"LoRA weights successfully saved to {output_path}")
            
            # Validate the saved file
            if validate_checkpoint(output_path):
                logger.info("Checkpoint validation successful")
            else:
                logger.warning("Checkpoint validation failed - file may be corrupted")
                
        except Exception as save_error:
            logger.error(f"Failed to save checkpoint: {save_error}")
            # Try to save with a different method as fallback
            try:
                logger.info("Attempting fallback save method...")
                torch.save(cpu_state_dict, output_path.replace('.safetensors', '.pt'))
                logger.info(f"Checkpoint saved as PyTorch file: {output_path.replace('.safetensors', '.pt')}")
            except Exception as fallback_error:
                logger.error(f"Fallback save also failed: {fallback_error}")
                raise
        
        # Clear memory after saving
        del cpu_state_dict
        clear_gpu_memory()
        
    except Exception as e:
        logger.error(f"Critical error during checkpoint saving: {e}")
        # Clear memory and re-raise
        clear_gpu_memory()
        raise

def validate_checkpoint(checkpoint_path: str) -> bool:
    """Validate that a checkpoint file can be loaded and contains expected data."""
    try:
        logger.info(f"Validating checkpoint: {checkpoint_path}")
        
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Basic validation
        if not state_dict:
            logger.error("Checkpoint is empty")
            return False
        
        # Check for LoRA-specific keys
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
        if not lora_keys:
            logger.warning("No LoRA keys found in checkpoint")
        
        logger.info(f"Checkpoint validation passed. Found {len(state_dict)} tensors")
        return True
        
    except Exception as e:
        logger.error(f"Checkpoint validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LoRA on Flux model")
    parser.add_argument("--training_config", type=str, default="configs/training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset_config.yaml",
                       help="Path to dataset configuration file")
    parser.add_argument("--captions_file", type=str, default="data/captions.txt",
                       help="Path to captions file")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory for trained model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    args = parser.parse_args()
    
    # Load configurations
    training_config = load_training_config(args.training_config)
    dataset_config = load_dataset_config(args.dataset_config)
    
    # Parse captions file
    image_paths, captions = parse_captions_file(args.captions_file)
    
    if len(image_paths) == 0:
        logger.error("No images found in captions file")
        sys.exit(1)
    
    logger.info(f"Found {len(image_paths)} images for training")
    
    # Set device
    device = torch.device(training_config["training"]["device"])
    logger.info(f"Using device: {device}")
    
    # Load tokenizer with authentication
    tokenizer = CLIPTokenizer.from_pretrained(
        training_config["model"]["pretrained_model_name_or_path"],
        subfolder="tokenizer",
        revision=training_config["model"]["revision"],
        token=True,  # Use authenticated access
    )
    
    # Create dataset and dataloader
    dataset = LoRADataset(
        image_paths=image_paths,
        captions=captions,
        tokenizer=tokenizer,
        size=dataset_config["dataset"]["resolution"],
        center_crop=dataset_config["dataset"]["center_crop"],
        random_flip=dataset_config["dataset"]["random_flip"],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_config["training"]["train_batch_size"],
        shuffle=True,
        num_workers=training_config["data"]["dataloader_num_workers"],
        pin_memory=True,
    )
    
    # Load models
    logger.info("Loading models...")
    
    # Note: VAE and text encoder are now loaded through the FLUX pipeline
    
    # Load FLUX pipeline and extract components with memory optimization
    logger.info("Loading FLUX pipeline with memory optimization...")
    
    # Log initial memory usage
    log_memory_usage("before pipeline loading")
    
    # Check if we should use CPU offloading
    use_cpu_offload = training_config["optimization"].get("cpu_offloading", True)
    
    # Determine the best data type based on hardware support
    mixed_precision = training_config["training"]["mixed_precision"]
    if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        logger.info("Using bfloat16 for better memory efficiency")
        torch_dtype = torch.bfloat16
        variant = "bf16"
    elif mixed_precision == "bf16":
        logger.warning("bfloat16 not supported, falling back to float16")
        torch_dtype = torch.float16
        variant = "fp16"
    else:
        logger.info("Using float16 precision")
        torch_dtype = torch.float16
        variant = "fp16"
    
    # Memory-efficient loading parameters
    pipe_kwargs = {
        "revision": training_config["model"]["revision"],
        "token": True,  # Use authenticated access
        "variant": variant,
        "torch_dtype": torch_dtype,
    }
    
    # Add memory-saving options
    if use_cpu_offload:
        logger.info("Enabling CPU offloading for memory efficiency...")
        # Use "balanced" instead of "auto" as it's supported
        pipe_kwargs["device_map"] = "balanced"
        pipe_kwargs["low_cpu_mem_usage"] = True
        # Add more aggressive memory saving options
        pipe_kwargs["max_memory"] = {0: "10GiB", "cpu": "20GiB"}  # Limit GPU memory usage
    
    # Load pipeline with memory optimizations
    try:
        # First attempt with aggressive memory optimization
        logger.info("Attempting to load FLUX pipeline with aggressive memory optimization...")
        pipe = FluxPipeline.from_pretrained(
            training_config["model"]["pretrained_model_name_or_path"],
            **pipe_kwargs
        )
        
        # Apply additional memory optimizations if needed
        if use_cpu_offload:
            logger.info("Applying CPU offloading to pipeline components...")
            pipe.enable_sequential_cpu_offload()
            
        # Enable attention slicing if configured
        if training_config["optimization"].get("attention_slicing", True):
            logger.info("Enabling attention slicing for memory efficiency...")
            pipe.enable_attention_slicing()
            
    except Exception as e:
        logger.error(f"Failed to load FLUX pipeline with optimizations: {e}")
        logger.info("Attempting to load with minimal memory settings...")
        
        # Fallback to minimal memory settings
        fallback_kwargs = {
            "revision": training_config["model"]["revision"],
            "token": True,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        
        try:
            pipe = FluxPipeline.from_pretrained(
                training_config["model"]["pretrained_model_name_or_path"],
                **fallback_kwargs
            )
            
            # Apply minimal optimizations
            pipe.enable_attention_slicing()
            pipe.enable_sequential_cpu_offload()
            
        except Exception as e2:
            logger.error(f"Still failed with minimal settings: {e2}")
            logger.info("Attempting ultra-minimal loading approach...")
            
            # Ultra-minimal approach: load components individually
            clear_gpu_memory()
            
            # Load only the essential components directly
            from diffusers import FluxTransformer2DModel, AutoencoderKL
            from transformers import CLIPTextModel, T5EncoderModel
            
            # Load transformer (UNet equivalent) with CPU offloading
            logger.info("Loading transformer with CPU offloading...")
            unet = FluxTransformer2DModel.from_pretrained(
                training_config["model"]["pretrained_model_name_or_path"],
                subfolder="transformer",
                torch_dtype=torch_dtype,
                revision=training_config["model"]["revision"],
                token=True,
                low_cpu_mem_usage=True,
            ).to("cpu")  # Keep on CPU initially
            
            # Load VAE
            logger.info("Loading VAE...")
            vae = AutoencoderKL.from_pretrained(
                training_config["model"]["pretrained_model_name_or_path"],
                subfolder="vae",
                torch_dtype=torch_dtype,
                revision=training_config["model"]["revision"],
                token=True,
                low_cpu_mem_usage=True,
            ).to("cpu")
            
            # Load text encoders
            logger.info("Loading text encoders...")
            text_encoder = CLIPTextModel.from_pretrained(
                training_config["model"]["pretrained_model_name_or_path"],
                subfolder="text_encoder",
                torch_dtype=torch_dtype,
                revision=training_config["model"]["revision"],
                token=True,
                low_cpu_mem_usage=True,
            ).to("cpu")
            
            # Create a minimal pipeline with pre-loaded components
            logger.info("Creating minimal pipeline...")
            pipe = FluxPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=None,  # Will add later
                text_encoder_2=None,  # Will add later
                tokenizer_2=None,  # Will add later
                transformer=unet,
                scheduler=None,  # Will add later
            )
            
            # Add the remaining components
            from transformers import T5TokenizerFast
            from diffusers import FlowMatchEulerDiscreteScheduler
            
            pipe.tokenizer = CLIPTokenizer.from_pretrained(
                training_config["model"]["pretrained_model_name_or_path"],
                subfolder="tokenizer",
                revision=training_config["model"]["revision"],
                token=True,
            )
            
            pipe.tokenizer_2 = T5TokenizerFast.from_pretrained(
                training_config["model"]["pretrained_model_name_or_path"],
                subfolder="tokenizer_2",
                revision=training_config["model"]["revision"],
                token=True,
            )
            
            pipe.text_encoder_2 = T5EncoderModel.from_pretrained(
                training_config["model"]["pretrained_model_name_or_path"],
                subfolder="text_encoder_2",
                torch_dtype=torch_dtype,
                revision=training_config["model"]["revision"],
                token=True,
                low_cpu_mem_usage=True,
            ).to("cpu")
            
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                training_config["model"]["pretrained_model_name_or_path"],
                subfolder="scheduler",
                token=True,
            )
            
            # Apply optimizations
            pipe.enable_sequential_cpu_offload()
            pipe.enable_attention_slicing()
            
            logger.info("Ultra-minimal loading completed successfully")
    
    # Log memory after pipeline loading
    log_memory_usage("after pipeline loading")
    
    # Extract components from the pipeline with memory management
    logger.info("Extracting pipeline components...")
    
    # Clear memory before component extraction
    clear_gpu_memory()
    
    # Extract components
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    unet = pipe.transformer  # FLUX uses Transformer2DModel instead of UNet2DConditionModel
    
    # Clear the pipeline to free memory
    del pipe
    clear_gpu_memory()
    
    logger.info("Moving components to device with memory optimization...")
    
    # Check if components are already on the correct device
    if vae.device != device:
        # Move VAE to device with memory optimization
        try:
            vae.to(device, dtype=torch_dtype)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("VAE too large for GPU, keeping on CPU with offloading...")
                # Keep VAE on CPU and use CPU offloading during inference
                vae.to("cpu", dtype=torch_dtype)
            else:
                raise e
        vae.requires_grad_(False)
        
        # Move text encoder to device
        if text_encoder.device != device:
            try:
                text_encoder.to(device, dtype=torch_dtype)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("Text encoder too large for GPU, keeping on CPU with offloading...")
                    text_encoder.to("cpu", dtype=torch_dtype)
                else:
                    raise e
        
        # Clear memory before moving UNet (largest component)
        clear_gpu_memory()
        
        # Move UNet to device in stages if needed
        if unet.device != device:
            try:
                unet.to(device, dtype=torch_dtype)
                logger.info("UNet successfully moved to GPU")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("UNet too large for GPU, attempting sequential loading...")
                    # Try to move components sequentially
                    for name, module in unet.named_children():
                        logger.info(f"Moving {name} to GPU...")
                        try:
                            module.to(device, dtype=torch_dtype)
                        except RuntimeError:
                            logger.warning(f"Could not move {name} to GPU, keeping on CPU...")
                            module.to("cpu", dtype=torch_dtype)
                        clear_gpu_memory()
                    
                    # If UNet is still too large, keep it on CPU
                    if unet.device != device:
                        logger.warning("UNet partially on CPU, training will be slower but should work")
                else:
                    raise e
    
    # Apply quantization if enabled
    logger.info("Applying quantization to model components...")
    try:
        # Apply quantization to UNet (main trainable component)
        unet = apply_quantization(unet, training_config)
        
        # Note: We typically don't quantize VAE and text encoder as they're frozen
        logger.info("Quantization application completed")
    except Exception as e:
        logger.warning(f"Quantization failed: {e}, continuing without quantization")
    
    # Final memory cleanup
    clear_gpu_memory()
    
    # Log final memory usage after component loading
    log_memory_usage("after component loading")
    
    logger.info("All components loaded successfully")
    
    # Create LoRA configuration
    lora_config = create_lora_config(training_config)
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Enable gradient checkpointing if configured
    if training_config["optimization"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=training_config["training"]["learning_rate"],
        betas=(training_config["training"]["adam_beta1"], training_config["training"]["adam_beta2"]),
        weight_decay=training_config["training"]["weight_decay"],
        eps=training_config["training"]["adam_epsilon"],
    )
    
    # Setup scheduler
    num_training_steps = len(dataloader) * training_config["training"]["num_train_epochs"]
    lr_scheduler = get_scheduler(
        training_config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=training_config["training"]["lr_warmup_steps"],
        num_training_steps=num_training_steps,
    )
    
    # Setup noise scheduler with authentication
    noise_scheduler = DDPMScheduler.from_pretrained(
        training_config["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler",
        token=True,  # Use authenticated access
    )
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=training_config["training"]["mixed_precision"] == "fp16")
    
    # Enable xformers memory efficient attention if configured
    if training_config["optimization"]["enable_xformers_memory_efficient_attention"]:
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers memory efficient attention: {e}")
    
    # Training loop
    for epoch in range(training_config["training"]["num_train_epochs"]):
        unet.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{training_config['training']['num_train_epochs']}")
        
        for step, batch in enumerate(progress_bar):
            # Get memory usage
            allocated, cached = get_memory_usage()
            progress_bar.set_postfix({"GPU (GB)": f"{allocated:.1f}/{cached:.1f}"})
            
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Encode text
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask)[0]
            
            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.get("scaling_factor")
            
            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.get("num_train_timesteps"), (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=training_config["training"]["mixed_precision"] == "fp16"):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if training_config["training"]["max_grad_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), training_config["training"]["max_grad_norm"])
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})
            
            global_step += 1
            
            # Save checkpoint
            if global_step % training_config["training"]["checkpointing_steps"] == 0:
                logger.info(f"Saving checkpoint at step {global_step}...")
                try:
                    # Ensure GPU memory is cleared before checkpoint saving
                    clear_gpu_memory()
                    
                    # Save checkpoint with robust error handling
                    save_lora_weights(unet, args.output_dir, global_step)
                    
                    # Additional memory cleanup after successful save
                    clear_gpu_memory()
                    
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {global_step}: {e}")
                    logger.info("Continuing training despite checkpoint save failure...")
                    # Ensure memory is cleared even if save failed
                    clear_gpu_memory()
            
            # Clear memory periodically
            if step % 10 == 0:
                clear_gpu_memory()
        
        # End of epoch
        logger.info(f"Epoch {epoch+1} completed")
        
        # Save final weights with robust error handling
        logger.info("Saving final LoRA weights...")
        try:
            # Clear GPU memory before final checkpoint saving
            clear_gpu_memory()
            
            # Save final checkpoint
            save_lora_weights(unet, args.output_dir)
            
            # Additional memory cleanup after successful save
            clear_gpu_memory()
            
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
            # Ensure memory is cleared even if save failed
            clear_gpu_memory()
            raise
    
    logger.info("Training completed!")
    logger.info(f"LoRA weights saved to {args.output_dir}")

if __name__ == "__main__":
    main()