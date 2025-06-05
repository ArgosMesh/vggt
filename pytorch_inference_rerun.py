# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import sys
import cv2
import rerun as rr
from PIL import Image
import tempfile
import os

sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cpu"

def preprocess_images_640x480_then_load(image_paths):
    """
    First resize all images to 640x480, then apply standard preprocessing.
    """
    # Create temporary files for resized images
    temp_paths = []
    for image_path in image_paths:
        # Open and resize to 640x480
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        
        # Resize to 640x480
        img_resized = img.resize((640, 480), Image.Resampling.BICUBIC)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        img_resized.save(temp_file.name)
        temp_paths.append(temp_file.name)
    
    # Apply standard preprocessing
    images = load_and_preprocess_images(temp_paths)
    
    # Clean up temporary files
    for temp_path in temp_paths:
        os.unlink(temp_path)
    
    return images

print("Initializing and loading VGGT PyTorch model...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)

# Define image list
image_paths = [
    "examples/gq/01.png",
    "examples/gq/02.png",
    "examples/gq/03.png",
    # "examples/gq/04.png",
    # "examples/gq/05.png",
    # "examples/gq/06.png",
    # "examples/gq/07.png",
    # "examples/gq/08.png"
]

print(f"Loading {len(image_paths)} images...")
images = preprocess_images_640x480_then_load(image_paths).to(device)
print(f"Preprocessed images shape: {images.shape}")

# Run PyTorch inference
print("Running PyTorch inference...")
dtype = torch.bfloat16

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

print("Converting pose encoding to extrinsic and intrinsic matrices...")
extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

# Remove batch dimension if present
if extrinsic.dim() == 4 and extrinsic.shape[0] == 1:
    extrinsic = extrinsic.squeeze(0)
if intrinsic.dim() == 4 and intrinsic.shape[0] == 1:
    intrinsic = intrinsic.squeeze(0)

predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic

# Generate world points from depth map
print("Computing world points from depth map...")
depth_map = predictions["depth"]  # (S, H, W, 1)

# The unproject function expects (S, H, W, 1) format and will squeeze(-1) internally
# So we need to ensure we have the right shape with the channel dimension
if depth_map.dim() == 5:  # (1, S, H, W, 1)
    depth_map_for_unproject = depth_map.squeeze(0)  # Remove batch dimension -> (S, H, W, 1)
elif depth_map.dim() == 3:  # (S, H, W) - add channel dimension
    depth_map_for_unproject = depth_map.unsqueeze(-1)  # -> (S, H, W, 1)
else:
    depth_map_for_unproject = depth_map

world_points = unproject_depth_map_to_point_map(depth_map_for_unproject, predictions["extrinsic"], predictions["intrinsic"])
predictions["world_points_from_depth"] = world_points

# Save depth images
print("Saving PyTorch depth images...")
depth_map_np = depth_map.cpu().numpy()

# Remove batch dimension and iterate through sequence dimension
if depth_map_np.ndim == 5:  # (1, S, H, W, 1)
    depth_map_np = depth_map_np.squeeze(0)  # (S, H, W, 1)

for i in range(depth_map_np.shape[0]):  # Iterate through sequence
    depth = depth_map_np[i]  # (H, W, 1)
    
    # Normalize depth to 0-255 range for visualization
    depth_normalized = depth.squeeze()  # Remove channel dimension -> (H, W)
    
    depth_min, depth_max = depth_normalized.min(), depth_normalized.max()
    if depth_max > depth_min:
        depth_vis = ((depth_normalized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth_normalized, dtype=np.uint8)
    
    # Apply colormap for better visualization
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
    
    # Save depth image
    depth_filename = f"pytorch_test_depth_{i:02d}.png"
    cv2.imwrite(depth_filename, depth_colored)
    print(f"Saved PyTorch depth image: {depth_filename}")

# Convert predictions to numpy for saving
predictions_np = {}
for key in predictions.keys():
    if isinstance(predictions[key], torch.Tensor):
        predictions_np[key] = predictions[key].cpu().numpy()
    else:
        predictions_np[key] = predictions[key]

# Save predictions
prediction_save_path = "pytorch_test_predictions.npz"
np.savez(prediction_save_path, **predictions_np)
print(f"Saved PyTorch predictions to: {prediction_save_path}")

# Visualize pointcloud with rerun
print("Visualizing PyTorch colored pointcloud with rerun...")
rr.init("VGGT_PyTorch_Pointcloud")
rr.spawn()

# Get world points and original images
world_points_vis = predictions_np["world_points_from_depth"]  # (S, H, W, 3)
original_images = images.cpu().numpy()  # (S, C, H, W)

# Flatten the world points from all frames
points_flattened = world_points_vis.reshape(-1, 3)

# Extract colors from original images
# Reshape images to match world points structure: (S, H, W, C)
colors_array = original_images.transpose(0, 2, 3, 1)  # (S, C, H, W) -> (S, H, W, C)
colors_flattened = colors_array.reshape(-1, 3)  # Flatten to match points

# Denormalize colors to [0, 255] range
colors_flattened = np.clip(colors_flattened, 0, 1)
colors_flattened = (colors_flattened * 255).astype(np.uint8)

# Remove invalid points (those with zero depth or extreme values)
valid_mask = np.all(np.isfinite(points_flattened), axis=1)
valid_mask &= np.linalg.norm(points_flattened, axis=1) < 100  # Remove points too far away

points_valid = points_flattened[valid_mask]
colors_valid = colors_flattened[valid_mask]

# Log the colored pointcloud
rr.log("world/pointcloud", rr.Points3D(points_valid, colors=colors_valid, radii=0.01))

# Also log original images for reference
for i in range(original_images.shape[0]):
    img = original_images[i].transpose(1, 2, 0)  # CHW to HWC
    # Denormalize image (assuming it was normalized to [-1, 1] or [0, 1])
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    rr.log(f"images/frame_{i:02d}", rr.Image(img))

# Log depth images for reference
for i in range(depth_map_np.shape[0]):
    depth_normalized = depth_map_np[i].squeeze()
    depth_min, depth_max = depth_normalized.min(), depth_normalized.max()
    if depth_max > depth_min:
        depth_vis = (depth_normalized - depth_min) / (depth_max - depth_min)
    else:
        depth_vis = np.zeros_like(depth_normalized)
    rr.log(f"depth/frame_{i:02d}", rr.DepthImage(depth_vis))

print(f"Visualized {len(points_valid)} colored points out of {len(points_flattened)} total points")
print("PyTorch inference and rerun visualization completed successfully!")
print(f"PyTorch depth images saved as: pytorch_test_depth_00.png, pytorch_test_depth_01.png, pytorch_test_depth_02.png")