# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import sys
import cv2
import onnxruntime as ort

sys.path.append("vggt/")

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cpu"

# Load ONNX model
print("Loading ONNX model...")
ort_session = ort.InferenceSession("vggt.onnx")

# Get model input/output info
input_info = ort_session.get_inputs()[0]
output_info = ort_session.get_outputs()

print(f"Model input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
print(f"Model outputs: {[out.name for out in output_info]}")

# Load first three kitchen images (same as onnx_export.py)
image_paths = [
    "examples/kitchen/images/00.png",
    "examples/kitchen/images/01.png", 
    "examples/kitchen/images/02.png"
]

print(f"Loading {len(image_paths)} kitchen images...")
images = load_and_preprocess_images(image_paths).to(device)
print(f"Preprocessed images shape: {images.shape}")

# Convert to numpy for ONNX
images_np = images.numpy()

# Run ONNX inference
print("Running ONNX inference...")
try:
    outputs = ort_session.run(None, {input_info.name: images_np})
    print("ONNX inference successful!")
    
    # Parse outputs based on the model structure
    predictions = {}
    for i, output in enumerate(outputs):
        output_name = output_info[i].name
        predictions[output_name] = output
    
    print(f"Prediction keys: {list(predictions.keys())}")
    
    # Try to identify depth and pose encoding from the output shapes
    depth_map = None
    pose_enc = None
    
    # Look for depth map based on expected shape (batch, frames, height, width, 1)
    for key, value in predictions.items():
        print(f"{key}: shape={value.shape}")
        if len(value.shape) == 5 and value.shape[-1] == 1:  # Likely depth
            print(f"Found potential depth map in {key}")
            depth_map = value
        elif len(value.shape) == 3 and value.shape[-1] == 9:  # Likely pose encoding
            print(f"Found potential pose encoding in {key}")
            pose_enc = value

except Exception as e:
    print(f"ONNX inference failed: {e}")
    sys.exit(1)

# Process depth predictions if found
if depth_map is not None:
    print(f"Processing depth map with shape: {depth_map.shape}")
    
    # Remove batch dimension if present
    if depth_map.ndim == 5:  # (1, S, H, W, 1)
        depth_map = depth_map.squeeze(0)
    elif depth_map.ndim == 4 and depth_map.shape[0] == 1:  # (1, H, W, 1)
        depth_map = depth_map.squeeze(0)

    
    # Save depth images
    print("Saving ONNX depth images...")
    for i, depth in enumerate(depth_map):
        # Normalize depth to 0-255 range for visualization
        depth_normalized = depth.squeeze()  # Remove channel dimension if present
        depth_min, depth_max = depth_normalized.min(), depth_normalized.max()
        
        if depth_max > depth_min:
            depth_vis = ((depth_normalized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth_normalized, dtype=np.uint8)
        
        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
        
        # Save depth image
        depth_filename = f"onnx_complete_test_depth_{i:02d}.png"
        cv2.imwrite(depth_filename, depth_colored)
        print(f"Saved ONNX depth image: {depth_filename}")
    
    # If pose encoding is available, compute camera parameters
    if pose_enc is not None:
        print("Computing camera parameters from pose encoding...")
        if pose_enc.ndim == 3 and pose_enc.shape[0] == 1:  # Remove batch dimension
            pose_enc = pose_enc.squeeze(0)
        
        # Convert to torch tensor for processing (the utility functions expect torch tensors)
        pose_enc_tensor = torch.from_numpy(pose_enc)
        image_shape = images.shape[-2:]
        
        try:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc_tensor, image_shape)
            
            # Convert back to numpy
            extrinsic_np = extrinsic.numpy()
            intrinsic_np = intrinsic.numpy()
            
            print(f"Extrinsic matrices shape: {extrinsic_np.shape}")
            print(f"Intrinsic matrices shape: {intrinsic_np.shape}")
            
            # Generate world points from depth map
            print("Computing world points from ONNX depth predictions...")
            world_points = unproject_depth_map_to_point_map(
                torch.from_numpy(depth_map), 
                torch.from_numpy(extrinsic_np), 
                torch.from_numpy(intrinsic_np)
            ).numpy()
            
            print(f"World points shape: {world_points.shape}")
            
            # Save all predictions
            onnx_predictions = {
                "depth": depth_map,
                "pose_enc": pose_enc,
                "extrinsic": extrinsic_np,
                "intrinsic": intrinsic_np,
                "world_points_from_depth": world_points
            }
            
        except Exception as e:
            print(f"Error in pose encoding processing: {e}")
            print("Skipping camera parameter computation, saving depth only")
            # Save predictions without camera parameters
            onnx_predictions = {
                "depth": depth_map,
                "pose_enc": pose_enc
            }
        
        # Add other predictions if available
        for key, value in predictions.items():
            if key not in ["depth", "pose_enc"]:  # Don't duplicate processed data
                onnx_predictions[f"raw_{key}"] = value
        
        prediction_save_path = "onnx_complete_test_predictions.npz"
        np.savez(prediction_save_path, **onnx_predictions)
        print(f"Saved ONNX predictions to: {prediction_save_path}")

else:
    print("Depth predictions not found in expected format")
    
    # Save raw outputs anyway
    raw_save_path = "onnx_complete_test_raw_outputs.npz"
    np.savez(raw_save_path, **predictions)
    print(f"Saved raw ONNX outputs to: {raw_save_path}")

print("ONNX model test completed successfully!")
print(f"ONNX depth images saved as: onnx_complete_test_depth_00.png, onnx_complete_test_depth_01.png, onnx_complete_test_depth_02.png")