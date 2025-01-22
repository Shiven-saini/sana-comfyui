# Author :- Shiven Saini
# Email :- shiven.career@proton.me

import torch
from safetensors.torch import save_file
import argparse
from pathlib import Path

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Convert a .pth model file to .safetensors format.')
parser.add_argument('input_file', type=str, help='Path to the input .pth file')
args = parser.parse_args()

# Determine input and output file paths
input_path = Path(args.input_file)
output_path = input_path.with_suffix('.safetensors')

# Load the .pth file
checkpoint = torch.load(str(input_path), map_location="cpu")

# Check if the file contains a 'state_dict' key
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    # Assume the checkpoint itself is the state_dict
    state_dict = checkpoint

# Ensure all values in the state_dict are tensors
for key, value in state_dict.items():
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Key `{key}` is invalid, expected torch.Tensor but received {type(value)}")

# Save as .safetensors
save_file(state_dict, str(output_path))

print(f"Conversion complete: Saved as {output_path}")
