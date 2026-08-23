"""
Step 1: Download the Raw, Unquantized Qwen2.5-Coder-3B Model from Hugging Face.
"""

import os
from huggingface_hub import snapshot_download

# Target model on Hugging Face (Raw unquantized FP16 / BF16 weights)
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"

# Local folder where the model files will be saved
SAVE_DIR = "models/raw_qwen2.5_3b"

def download_model():
    print("=" * 60)
    print(f"[*] Downloading raw model: {MODEL_ID}")
    print(f"[*] Destination directory: {SAVE_DIR}")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Download model weights, tokenizer, and config files
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=SAVE_DIR
    )

    print("\n" + "=" * 60)
    print(f"[✓] Successfully downloaded raw model to: {path}")
    print("=" * 60)

if __name__ == "__main__":
    download_model()
