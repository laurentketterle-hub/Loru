#!/usr/bin/env python3
"""Detect near-duplicate gloss frames in Loru sign packs."""
import json, os, sys, hashlib
from pathlib import Path

def hash_offsets(json_path, precision=2):
    """Generate hash from frame offsets for duplicate detection."""
    with open(json_path) as f:
        data = json.load(f)
    
    frames = data.get("frames", [])
    offsets = []
    for frame in frames:
        for joint in frame.get("joints", []):
            x = round(joint.get("x", 0), precision)
            y = round(joint.get("y", 0), precision)
            offsets.append(f"{x},{y}")
    
    return hashlib.sha256(",".join(offsets).encode()).hexdigest(), len(offsets)

def find_duplicates(data_dir="data/sign-packs"):
    """Find near-duplicate gloss frames across all packs."""
    hashes = {}
    for json_file in Path(data_dir).rglob("*.json"):
        h, n_offsets = hash_offsets(str(json_file))
        if h in hashes:
            print(f"DUPLICATE: {json_file} = {hashes[h]['file']} ({n_offsets} offsets)")
        else:
            hashes[h] = {"file": str(json_file), "offsets": n_offsets}
    print(f"Checked {len(hashes)} unique hashes")

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sign-packs"
    find_duplicates(data_dir)
