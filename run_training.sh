#!/bin/bash

# LoRA Fine-tuning Script for Flux Model
# This script automates the training process for 20 images

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}  LoRA Fine-tuning with Flux Model        ${NC}"
echo -e "${BLUE}  Optimized for NVIDIA RTX 1080 GPU      ${NC}"
echo -e "${BLUE}===========================================${NC}"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3."
    exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "nvidia-smi not found. Make sure CUDA is installed and GPU drivers are up to date."
fi

# Install dependencies
print_info "Installing dependencies..."
pip3 install -r requirements.txt

# Check if data directory exists
if [ ! -d "data/images" ]; then
    print_info "Creating data/images directory..."
    mkdir -p data/images
fi

# Check if images directory has files
image_count=$(find data/images -name "*.webp" | wc -l)
if [ $image_count -eq 0 ]; then
    print_warning "No .webp images found in data/images directory."
    print_info "Please place your 20 training images in the data/images directory."
    print_info "Images should be in .webp format."
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Check if captions file exists
if [ ! -f "data/captions.txt" ]; then
    print_info "Creating sample captions file..."
    # The captions.txt file already exists, so we don't need to create it
fi

# Check system information
print_info "Checking system information..."
python3 train.py --system_info

# Validate dataset
print_info "Validating dataset..."
if python3 train.py --validate_only; then
    print_success "Dataset validation passed!"
else
    print_error "Dataset validation failed. Please check your data."
    exit 1
fi

# Dry run to show what would happen
print_info "Performing dry run..."
python3 train.py --dry_run

# Ask for confirmation
echo ""
print_warning "Training will start now. This may take several hours."
print_warning "Make sure you have sufficient disk space and your GPU is not overheating."
read -p "Do you want to continue? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Training cancelled."
    exit 0
fi

# Start training
print_info "Starting LoRA training..."
print_info "Logs will be saved to logs/training.log"
print_info "Checkpoints will be saved to output/"
print_info "Press Ctrl+C to stop training early."

# Run training
if python3 train.py; then
    print_success "Training completed successfully!"
    print_success "Your LoRA weights are saved in output/lora_weights_final.safetensors"
    
    # Show sample prompts
    echo ""
    print_info "Sample prompts for testing your LoRA:"
    python3 -c "
from scripts.utils import create_sample_prompts
prompts = create_sample_prompts()
for i, prompt in enumerate(prompts, 1):
    print(f'{i}. {prompt}')
"
    
    echo ""
    print_success "You can now use the generated .safetensors file with inference tools!"
else
    print_error "Training failed. Check logs/training.log for details."
    exit 1
fi