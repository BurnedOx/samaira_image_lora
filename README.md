# LoRA Fine-tuning with Flux Model

This project provides a complete solution for fine-tuning a LoRA (Low-Rank Adaptation) adapter on the Flux model using 20 input images. The trained LoRA will be saved as a `.safetensors` file that can be used with various inference tools.

## Features

- Fine-tune LoRA on Flux model with 20 images
- Optimized for NVIDIA RTX 1080 GPU (8GB VRAM)
- Memory-efficient training with mixed precision
- Automatic generation of `.safetensors` file
- Comprehensive logging and monitoring
- Configurable training parameters

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with at least 8GB VRAM (RTX 1080 or recommended)
- At least 16GB RAM
- 50GB free disk space (for models and checkpoints)

### Software Requirements
- Python 3.8 or higher
- CUDA 11.8 or higher
- PyTorch with CUDA support

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
samaira_image_lora/
├── data/
│   ├── images/           # Place your 20 .webp images here
│   └── captions.txt      # Image captions configuration
├── configs/
│   ├── dataset_config.yaml     # Dataset configuration
│   └── training_config.yaml    # Training parameters
├── scripts/
│   ├── train_lora.py          # Main training script
│   ├── generate_safetensors.py  # Convert to safetensors
│   └── utils.py               # Utility functions
├── output/                   # Training outputs and checkpoints
├── logs/                     # Training logs
├── requirements.txt          # Python dependencies
├── train.py                  # Main entry point
└── README.md                 # This file
```

## Setup Instructions

### 1. Prepare Your Images

Place your 20 training images in the `data/images/` directory. The images should be in `.webp` format as specified.

```bash
# Copy your images to the data directory
cp /path/to/your/images/*.webp data/images/
```

### 2. Configure Captions

Edit `data/captions.txt` to match your image filenames and add appropriate captions. The format is:

```
image_filename: caption
```

Replace `[your_subject]` with the actual subject you want to train (e.g., "my cat", "a specific person", etc.).

Example:
```
image_001.webp: a photo of my cat
image_002.webp: my cat sleeping on the couch
```

### 3. Configure Training Parameters

Edit `configs/training_config.yaml` to adjust training parameters if needed. The default configuration is optimized for RTX 1080:

- **LoRA rank**: 16
- **Batch size**: 1
- **Learning rate**: 1e-4
- **Mixed precision**: fp16
- **Gradient accumulation**: 4 steps

## Training

### Quick Start

Run the training with default settings:

```bash
python train.py
```

### Advanced Usage

```bash
# Validate dataset only
python train.py --validate_only

# Check system information
python train.py --system_info

# Dry run to see what would happen
python train.py --dry_run

# Custom training parameters
python train.py --epochs 150 --batch_size 1 --learning_rate 5e-5

# Custom output directory
python train.py --output_dir /path/to/custom/output
```

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data_dir` | Directory containing training images | `data/images` |
| `--captions_file` | Path to captions file | `data/captions.txt` |
| `--output_dir` | Output directory for trained model | `output` |
| `--epochs` | Number of training epochs | 100 |
| `--batch_size` | Training batch size | 1 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--rank` | LoRA rank | 16 |
| `--validate_only` | Only validate dataset | False |
| `--system_info` | Print system information | False |
| `--dry_run` | Show what would be done | False |

## Monitoring Training

### Logs

Training logs are saved to `logs/training.log` and also displayed in the console.

### GPU Memory Usage

The script monitors GPU memory usage and displays it in the progress bar:

```
Epoch 1/100: 100%|██████████| 20/20 [00:45<00:00,  2.25s/step, loss=0.1234, lr=1.00e-4, GPU (GB): 5.2/8.0]
```

### Checkpoints

Checkpoints are saved every 500 steps to `output/lora_weights_step_XXXX.safetensors`.

## Output

After training completes, you will find:

1. **Final LoRA weights**: `output/lora_weights_final.safetensors`
2. **Training configuration**: `output/training_config.yaml` and `output/training_config.json`
3. **Checkpoints**: Various `lora_weights_step_XXXX.safetensors` files
4. **Logs**: `logs/training.log`

## Using the Trained LoRA

The generated `.safetensors` file can be used with various inference tools that support LoRA adapters for the Flux model.

### Sample Prompts

After training, you can use prompts like:

```
a photo of [your_subject]
[your_subject] in a garden
a close-up photo of [your_subject]
artistic portrait of [your_subject]
```

Replace `[your_subject]` with whatever you used in your training captions.

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size to 1
   - Enable gradient checkpointing
   - Use mixed precision (fp16)
   - Reduce image resolution

2. **CUDA Error**
   - Check CUDA installation: `nvidia-smi`
   - Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
   - Update GPU drivers

3. **Import Errors**
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python version (requires 3.8+)

4. **Image Loading Errors**
   - Verify all images are in `data/images/`
   - Check image format (should be .webp)
   - Verify captions file format

### Performance Optimization

For RTX 1080 (8GB VRAM), the following optimizations are applied:

- Mixed precision training (fp16)
- Gradient checkpointing
- XFormers memory-efficient attention
- Gradient accumulation (4 steps)
- Small batch size (1)
- Attention slicing

## Configuration Files

### Dataset Configuration (`configs/dataset_config.yaml`)

```yaml
dataset:
  name: "custom_lora_dataset"
  path: "data/images"
  image_size: 512
  center_crop: true
  random_flip: true
  resolution: 512
  batch_size: 1
  num_workers: 4
  num_images: 20
```

### Training Configuration (`configs/training_config.yaml`)

Key sections include model settings, training parameters, optimization settings, and logging configuration.

## Memory Usage

Expected memory usage on RTX 1080:
- Model loading: ~3GB
- Training: ~5-6GB
- Peak usage: ~7GB

## Training Time

Estimated training time on RTX 1080:
- 20 images, 100 epochs: ~2-3 hours
- Actual time may vary based on image size and complexity

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure you have the right to use the Flux model and your training images.

## Support

If you encounter issues:
1. Check the troubleshooting section
2. Review the logs in `logs/training.log`
3. Verify your setup meets the prerequisites
4. Check GPU memory usage with `nvidia-smi`

## Acknowledgments

- Flux model by Black Forest Labs
- LoRA implementation by PEFT library
- Diffusers library by Hugging Face