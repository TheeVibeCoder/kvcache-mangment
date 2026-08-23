"""
====================================================================================================
VERSION 1: Inspection tool for viewing internal structure & tensor values of .safetensors files.
====================================================================================================
"""

import sys
import os
import mlx.core as mx
import numpy as np

def inspect_safetensors(filepath: str):
    if not os.path.exists(filepath):
        print(f"[!] File not found: {filepath}")
        return

    print("=" * 80)
    print(f"📦 SAFETENSORS FILE INSPECTOR: {os.path.basename(filepath)}")
    print("=" * 80)
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"📁 File Path : {filepath}")
    print(f"💾 File Size : {file_size_mb:.2f} MB ({os.path.getsize(filepath):,} bytes)")

    # Load tensors
    tensors = mx.load(filepath)
    print(f"🔑 Total Tensors Saved: {len(tensors)}")
    print("-" * 80)

    # 1. Metadata Inspection
    print("📋 1. METADATA & CONFIGURATION HEADER:")
    for k in ["_meta_bits", "_meta_group_size", "_meta_is_quantized"]:
        if k in tensors:
            val = tensors[k].item()
            print(f"   • {k:<25} = {val}")

    # 2. Summary by Layer
    print("\n📋 2. TENSOR KEYS BREAKDOWN (Sample Layer 0 vs Layer 35):")
    for layer_idx in [0, 1, 35]:
        print(f"\n   --- Layer {layer_idx} ---")
        layer_keys = [k for k in tensors.keys() if f"layer_{layer_idx}_" in k]
        for key in sorted(layer_keys):
            t = tensors[key]
            print(f"   • {key:<20} | Shape: {str(t.shape):<18} | Dtype: {str(t.dtype):<8} | Elements: {t.size:,}")

    # 3. Inside the Actual Tensor Values
    print("\n" + "=" * 80)
    print("🔍 3. DEEP DIVE: ACTUAL TENSOR VALUES INSIDE LAYER 0")
    print("=" * 80)

    if "layer_0_k_arr" in tensors:
        print("\n   [A] Keys Quantized Array (layer_0_k_arr) - Raw Packed Uint32 bits:")
        k_arr = np.array(tensors["layer_0_k_arr"])
        print(f"       Shape: {k_arr.shape}")
        print("       First 4 packed integers (hex format showing packed 8-bit weights):")
        first_vals = k_arr[0, 0, 0, :4]
        for i, val in enumerate(first_vals):
            print(f"         - Element [{i}]: {val} (Hex: 0x{val:08X})")

        print("\n   [B] Keys Scale Multiplier (layer_0_k_sc) - Float16 FP values:")
        k_sc = np.array(tensors["layer_0_k_sc"])
        print(f"       Shape: {k_sc.shape}")
        print(f"       Values: {k_sc[0, 0, 0, :]}")

        print("\n   [C] Keys Bias Offset (layer_0_k_bi) - Float16 FP values:")
        k_bi = np.array(tensors["layer_0_k_bi"])
        print(f"       Shape: {k_bi.shape}")
        print(f"       Values: {k_bi[0, 0, 0, :]}")

    if "layer_0_offset" in tensors:
        offset = tensors["layer_0_offset"].item()
        print(f"\n   [D] Layer Offset (Number of Tokens Cached): {offset} tokens")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "kv_cache_store/demo_code_session.safetensors"
    inspect_safetensors(target)
