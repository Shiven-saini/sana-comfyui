# Author :- Shiven Saini
# Email :- shiven.career@proton.me
# Modified to fetch 1024px res model

import os
from huggingface_hub import hf_hub_download, snapshot_download, HfFileSystem

def create_directories():
    """Create required directory structure with verification"""
    dirs = [
        "models/checkpoints",
        "models/vae",
        "models/text_encoders"
    ]
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            print(f"📁 Verified/Created directory: {d}")
        except Exception as e:
            print(f"❌ Failed to create directory {d}: {str(e)}")
            raise

def verify_file(repo_id, filename):
    """Verify file exists in repository before download"""
    fs = HfFileSystem()
    full_path = f"{repo_id}/{filename}"
    try:
        if fs.exists(full_path):
            print(f"✅ Verified file exists: {filename}")
            return True
        print(f"❌ File not found: {filename}")
        return False
    except Exception as e:
        print(f"🔴 Verification error: {str(e)}")
        return False

def download_model(repo_id, filename, local_dir, rename=None):
    """Generic download function with path correction"""
    try:
        if not verify_file(repo_id, filename):
            return False

        # Fix: Prevent nested directory creation
        adjusted_local_dir = os.path.dirname(os.path.join(local_dir, filename))
        os.makedirs(adjusted_local_dir, exist_ok=True)

        print(f"⬇️ Downloading {filename} from {repo_id}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,  # Base directory without subfolder
            etag_timeout=100,
            repo_type="model",
        )

        if rename:
            target_path = os.path.join(local_dir, rename)
            os.rename(path, target_path)
            print(f"🔀 Renamed to: {rename}")
            path = target_path

        print(f"✅ Successfully saved to: {path}")
        return True
    except Exception as e:
        print(f"🔥 Download failed: {str(e)}")
        return False

def main():
    print("\n🚀 Starting Model Download Process\n")
    create_directories()

    # --- Step 1: Download Sana 600M Checkpoint ---
    print("\n" + "="*40)
    print("Step 1: Downloading Sana 600M Model")
    print("="*40)
    success_1 = download_model(
        repo_id="Efficient-Large-Model/Sana_600M_1024px",
        filename="checkpoints/Sana_600M_1024px_MultiLing.pth",
        local_dir="models",  # Changed to base directory
        rename="checkpoints/Sana_600M_1024px_MultiLing.pth"  # Explicit path
    )

    # --- Step 2: Download VAE Model ---
    print("\n" + "="*40)
    print("Step 2: Downloading VAE Model")
    print("="*40)
    success_2 = download_model(
        repo_id="mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",
        filename="diffusion_pytorch_model.safetensors",
        local_dir="models/vae",
        rename="dc-ae-f32c32.safetensors"
    )

    # --- Step 3: Download Text Encoder ---
    print("\n" + "="*40)
    print("Step 3: Downloading Text Encoder")
    print("="*40)
    success_3 = False
    try:
        text_encoder_path = snapshot_download(
            repo_id="unsloth/gemma-2-2b-it-bnb-4bit",
            local_dir="models/text_encoders/gemma-2-2b-it-bnb-4bit",
            repo_type="model",
            resume_download=True,
            etag_timeout=100
        )
        print(f"✅ Text encoder saved to: {text_encoder_path}")
        success_3 = True
    except Exception as e:
        print(f"🔥 Text encoder download failed: {str(e)}")

    # --- Final Report ---
    print("\n" + "="*40)
    print("Download Summary")
    print("="*40)
    print(f"1. Sana 600M Checkpoint: {'✅ SUCCESS' if success_1 else '❌ FAILED'}")
    print(f"2. VAE Model:           {'✅ SUCCESS' if success_2 else '❌ FAILED'}")
    print(f"3. Text Encoder:        {'✅ SUCCESS' if success_3 else '❌ FAILED'}")
    print("\n" + "="*40)

if __name__ == "__main__":
    main()
