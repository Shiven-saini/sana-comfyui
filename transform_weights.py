# Author :- Shiven Saini
# Email :- shiven.career@proton.me

import torch
from safetensors.torch import save_file
from pathlib import Path
import sys

# Configuration
CHECKPOINT_DIR = Path("models/checkpoints")
EMOJI = {
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "processing": "⏳",
    "deleted": "🗑️",
    "files": "📁",
    "conversion": "⚙️",
    "security": "🛡️"
}

def main():
    print(f"\n{EMOJI['files']}  Scanning for .pth files in {CHECKPOINT_DIR}...")
    
    # Find all .pth files
    pth_files = list(CHECKPOINT_DIR.glob("*.pth"))
    
    if not pth_files:
        print(f"{EMOJI['error']} No .pth files found in directory")
        return
    
    print(f"{EMOJI['success']} Found {len(pth_files)} .pth file(s):")
    for idx, file in enumerate(pth_files, 1):
        print(f"  {idx}. {file.name}")
    
    # Process files
    converted = []
    for pth_file in pth_files:
        try:
            print(f"\n{EMOJI['processing']} Processing {pth_file.name}...")
            
            # Load checkpoint with security restrictions
            try:
                checkpoint = torch.load(
                    pth_file,
                    map_location="cpu",
                    weights_only=True  # Security-enabled loading
                )
            except RuntimeError as e:
                print(f"{EMOJI['error']} Security load failed: {str(e)}")
                print(f"{EMOJI['warning']} File might contain non-tensor data")
                print(f"{EMOJI['security']} Tip: Only use trusted models!")
                continue
            
            # Extract state_dict
            state_dict = checkpoint.get("state_dict", checkpoint)
            
            # Validate tensors
            tensor_errors = []
            for key, value in state_dict.items():
                if not isinstance(value, torch.Tensor):
                    tensor_errors.append(f"Key `{key}`: {type(value).__name__}")
            
            if tensor_errors:
                print(f"{EMOJI['error']} Non-tensor values found:")
                for error in tensor_errors[:3]:  # Show first 3 errors
                    print(f"  • {error}")
                if len(tensor_errors) > 3:
                    print(f"  ...and {len(tensor_errors)-3} more")
                continue
            
            # Create output path
            safetensors_path = pth_file.with_suffix(".safetensors")
            
            # Save converted file
            save_file(state_dict, str(safetensors_path))
            converted.append(pth_file)
            print(f"{EMOJI['conversion']} Successfully converted to {safetensors_path.name}")
            
        except Exception as e:
            print(f"{EMOJI['error']} Critical error processing {pth_file.name}:")
            print(f"  {str(e)}")
            continue

    # Delete original files prompt
    if converted:
        print(f"\n{EMOJI['warning']}  Conversion summary:")
        print(f"  • Successfully converted: {len(converted)} files")
        print(f"  • Failed conversions: {len(pth_files)-len(converted)} files")
        
        while True:
            response = input(
                f"\n{EMOJI['warning']} Delete original .pth files? (y/n): "
            ).lower()
            
            if response in ("y", "n"):
                break
            print(f"{EMOJI['error']} Please enter 'y' or 'n'")

        if response == "y":
            print(f"\n{EMOJI['deleted']} Deleting original files:")
            for file in converted:
                try:
                    file.unlink()
                    print(f"  Successfully deleted {file.name}")
                except Exception as e:
                    print(f"{EMOJI['error']} Failed to delete {file.name}: {str(e)}")

    print(f"\n{EMOJI['success']} Conversion process completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{EMOJI['error']} Operation cancelled by user")
        sys.exit(1)