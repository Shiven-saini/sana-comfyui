# Author :- Shiven Saini
# Email :- shiven.career@proton.me

import os
import shutil
from huggingface_hub import HfApi, snapshot_download, hf_hub_download

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")

def verify_file(file_path):
    """Check if file is actual content or LFS pointer"""
    with open(file_path, 'rb') as f:
        header = f.read(100)
        if b'version https://git-lfs.github.com' in header:
            raise ValueError(f"LFS pointer detected in {file_path}!")
    print(f"Verified: {os.path.basename(file_path)} is actual file ({os.path.getsize(file_path)/1e6:.1f}MB)")

def main():
    api = HfApi()

    # Step 1: Download text encoder model
    print("\nStep 1: Downloading text encoder model...")
    text_encoder_repo = "unsloth/gemma-2-2b-it-bnb-4bit"
    text_encoder_path = os.path.join("models", "text_encoders", "gemma-2-2b-it-bnb-4bit")
    create_directory(text_encoder_path)
    
    snapshot_download(
        repo_id=text_encoder_repo,
        local_dir=text_encoder_path,
        repo_type="model",
        force_download=True,
    )
    print(f"Downloaded text encoder model to: {text_encoder_path}")

    # Step 2: Download checkpoint
    print("\nStep 2: Downloading checkpoint...")
    checkpoint_repo = "Efficient-Large-Model/Sana_600M_512px"
    checkpoint_path = os.path.join("models", "checkpoints")
    create_directory(checkpoint_path)
    
    files = api.list_repo_files(repo_id=checkpoint_repo, repo_type="model")
    pth_file = next(
        (f for f in files 
         if f.startswith("checkpoints/") and f.endswith(".pth")),
        None
    )
    
    if not pth_file:
        raise FileNotFoundError("No .pth file found in checkpoints folder")
    
    downloaded_file = hf_hub_download(
        repo_id=checkpoint_repo,
        filename=pth_file,
        force_download=True,
    )
    target_file = os.path.join(checkpoint_path, os.path.basename(pth_file))
    shutil.move(downloaded_file, target_file)
    verify_file(target_file)

    # Step 3: Download VAE
    print("\nStep 3: Downloading VAE model...")
    vae_repo = "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"
    vae_path = os.path.join("models", "vae")
    create_directory(vae_path)
    
    files = api.list_repo_files(repo_id=vae_repo, repo_type="model")
    safetensors_file = next((f for f in files if f.endswith(".safetensors")), None)
    
    if not safetensors_file:
        raise FileNotFoundError("No .safetensors file found in repository")
    
    downloaded_file = hf_hub_download(
        repo_id=vae_repo,
        filename=safetensors_file,
        force_download=True,
    )
    target_file = os.path.join(vae_path, "dc-ae-f32c32-sana.safetensors")
    shutil.move(downloaded_file, target_file)
    verify_file(target_file)

    print("\nAll downloads completed and verified successfully!")

if __name__ == "__main__":
    main()