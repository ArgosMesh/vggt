# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import sys
from PIL import Image
import cv2

sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)

# Load first three kitchen images
image_paths = [
    "examples/kitchen/images/00.png",
    "examples/kitchen/images/01.png", 
    "examples/kitchen/images/02.png"
]

print(f"Loading {len(image_paths)} kitchen images...")
images = load_and_preprocess_images(image_paths).to(device)
print(f"Preprocessed images shape: {images.shape}")

# Run inference
print("Running inference...")
dtype = torch.bfloat16 #if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

print("Converting pose encoding to extrinsic and intrinsic matrices...")
extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic

# Convert tensors to numpy and remove batch dimension
for key in predictions.keys():
    if isinstance(predictions[key], torch.Tensor):
        predictions[key] = predictions[key].cpu().numpy().squeeze(0)

# Generate world points from depth map
print("Computing world points from depth map...")
depth_map = predictions["depth"]  # (S, H, W, 1)
world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
predictions["world_points_from_depth"] = world_points

# Save depth images
print("Saving depth images...")
for i, depth in enumerate(depth_map):
    # Normalize depth to 0-255 range for visualization
    depth_normalized = depth.squeeze()  # Remove channel dimension
    depth_min, depth_max = depth_normalized.min(), depth_normalized.max()
    if depth_max > depth_min:
        depth_vis = ((depth_normalized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth_normalized, dtype=np.uint8)
    
    # Apply colormap for better visualization
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
    
    # Save depth image
    depth_filename = f"depth_image_{i:02d}.png"
    cv2.imwrite(depth_filename, depth_colored)
    print(f"Saved depth image: {depth_filename}")

# Save predictions
prediction_save_path = "predictions.npz"
np.savez(prediction_save_path, **predictions)
print(f"Saved predictions to: {prediction_save_path}")

# Export to ONNX
print("Exporting model to ONNX...")
torch.onnx.export(
    model, 
    images, 
    "vggt.onnx", 
    verbose=True, 
    dynamo=True, 
    report=True, 
    opset_version=17,
    input_names=["input"], 
    output_names=["output"]
)

print("ONNX export completed successfully!")
print(f"Model exported to: vggt.onnx")
print(f"Depth images saved as: depth_image_00.png, depth_image_01.png, depth_image_02.png")