# Sana ComfyUI

## 🖥️ System Requirements

- **Minimum Hardware:**
  - NVIDIA Graphics Card with **4GB VRAM** minimum (for little-variant 512px variant)
  - NVIDIA Graphics Card with **6GB VRAM** minimum (for large-variant 1024px variant)
  
- **Software Requirements:**
  - Python 3.11 (⚠️ **Only** 3.11 is supported - other versions will not work)
  - CUDA Toolkit

## ⚙️ Installation & Setup

### 0. Clone the Repository
```bash
# Clone the repository
git clone https://github.com/Shiven-saini/sana-comfyui.git

# For little-variant 512px, switch to branch 'little'
git switch little

# For large-variant 1024px, switch to branch 'large'
git swtich large
```

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv env

# Activate environment (Linux/macOS)
source env/bin/activate

# For Windows use:
# .\env\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project requirements
pip install -r requirements.txt
```

### 3. Download Models
```bash
# Run model download script
python fetch_weights.py
```

📌 Important: You must download the models before running the application. This script will automatically download & setup these weights :
- Gemma Text Encoder [Hugging Face 🤗](https://huggingface.co/unsloth/gemma-2-2b-it-bnb-4bit)
- MIT DC-AE Encoder [Hugging Face 🤗](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers)
- SANA Model [Hugging Face 🤗](https://huggingface.co/collections/Efficient-Large-Model/sana-673efba2a57ed99843f11f9e)

### 4. Running the Application 

Before running:
- [x] Ensure virtual environment is activated
- [x] Verify models are downloaded (weights/ directory exists with all files)

To launch the local server :-
```bash
python main.py
```

#### Optional - Convert the weights .pth to .safetensors
```bash
python transform_weights.py
```
Make sure to select the newly generated .safetensors file in the comfyui node workflow!

ℹ️ Additional Information

- For optimal performance, close other GPU-intensive applications before running
- First run may take longer due to model initialization
- Monitor VRAM usage using nvidia-smi (Linux/Windows) or GPU monitoring tools like Nvtop on Linux.
