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
import rerun as rr

sys.path.append("vggt/")

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cpu"

# Load ONNX models
print("Loading split ONNX models...")
image_encoder_session = ort.InferenceSession("vggt_image_encoder.onnx")
aggregator_session = ort.InferenceSession("vggt_aggregator.onnx")

# Get model input/output info
encoder_input_info = image_encoder_session.get_inputs()[0]
encoder_output_info = image_encoder_session.get_outputs()[0]

aggregator_input_info = aggregator_session.get_inputs()
aggregator_output_info = aggregator_session.get_outputs()

print("="*60)
print("ONNX MODEL SPECIFICATIONS:")
print("="*60)
print(f"Image Encoder ONNX:")
print(f"  Input: {encoder_input_info.name}, shape: {encoder_input_info.shape}, type: {encoder_input_info.type}")
print(f"  Output: {encoder_output_info.name}, shape: {encoder_output_info.shape}, type: {encoder_output_info.type}")

print(f"\nAggregator ONNX:")
print(f"  Inputs:")
for inp in aggregator_input_info:
    print(f"    {inp.name}: shape={inp.shape}, type={inp.type}")
print(f"  Outputs:")
for out in aggregator_output_info:
    print(f"    {out.name}: shape={out.shape}, type={out.type}")
print("="*60)

# Load first three kitchen images (same as other test scripts)
image_paths = [
    # "examples/room/images/no_overlap_4.jpg",
    # "examples/room/images/no_overlap_2.jpg",
    # "examples/room/images/no_overlap_3.jpg",
    "examples/gq/01.png",
    "examples/gq/02.png",
    #"examples/gq/03.png",
    # "examples/gq/04.png",
    #"examples/gq/05.png",
    # "examples/gq/06.png",
    # "examples/gq/07.png",
    # "examples/gq/08.png"
]

def preprocess_images_640x480_then_load(image_paths):
    """
    First resize all images to 640x480, then apply standard preprocessing.
    """
    from PIL import Image
    import tempfile
    import os

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

print(f"Loading {len(image_paths)} kitchen images...")
images = preprocess_images_640x480_then_load(image_paths).to(device)
print(f"Preprocessed images shape: {images.shape}")

S, C_in, H, W = images.shape
B = 1  # Single batch for demo

# Add batch dimension if needed
if len(images.shape) == 4:
    images = images.unsqueeze(0)  # [1, S, 3, H, W]

B, S, C_in, H, W = images.shape

# Convert to numpy for ONNX
images_np = images.numpy()

# Run split ONNX inference
print("\n" + "="*60)
print("ACTUAL RUNTIME DIMENSIONS:")
print("="*60)
print("Running Image Encoder ONNX inference...")
try:
    # Process each frame separately and concatenate tokens
    patch_tokens_list = []
    for i in range(S):
        single_frame = images_np[:, i]  # [B, 3, H, W]
        print(f"Frame {i} input shape to Image Encoder: {single_frame.shape}")
        encoder_outputs = image_encoder_session.run(None, {encoder_input_info.name: single_frame})
        frame_tokens = encoder_outputs[0]  # [B, P, C]
        print(f"Frame {i} output shape from Image Encoder: {frame_tokens.shape}")
        patch_tokens_list.append(frame_tokens)

    # Concatenate all frame tokens in batch dimension
    patch_tokens = np.concatenate(patch_tokens_list, axis=0)  # [B*S, P, C]
    print(f"\nConcatenated patch tokens shape: {patch_tokens.shape}")

    print("Running Aggregator ONNX inference...")

    # Prepare aggregator inputs - check what inputs the model actually expects
    print(f"Preparing aggregator inputs:")
    aggregator_inputs = {}
    for inp in aggregator_input_info:
        if inp.name == "patch_tokens":
            aggregator_inputs["patch_tokens"] = patch_tokens
            print(f"  {inp.name}: {patch_tokens.shape}")
        elif inp.name == "sequence_length":
            aggregator_inputs["sequence_length"] = np.array([S], dtype=np.int64)
            print(f"  {inp.name}: {S}")
        elif inp.name == "height":
            aggregator_inputs["height"] = np.array([H], dtype=np.int64)
            print(f"  {inp.name}: {H}")
        elif inp.name == "width":
            aggregator_inputs["width"] = np.array([W], dtype=np.int64)
            print(f"  {inp.name}: {W}")

    aggregator_outputs = aggregator_session.run(None, aggregator_inputs)
    print("Split ONNX inference successful!")

    # Parse outputs based on the model structure
    predictions = {}

    print(f"\nONNX Output Order Check:")
    print(f"ONNX Model Output Names (in order):")
    for i, out_info in enumerate(aggregator_output_info):
        print(f"  {i}: {out_info.name}")

    print(f"Actual Runtime Outputs (in order):")
    for i, output in enumerate(aggregator_outputs):
        output_name = aggregator_output_info[i].name
        predictions[output_name] = output
        print(f"  {i}: {output_name} -> shape {output.shape}")

    print(f"\nAggregator output keys: {list(predictions.keys())}")

    # Process outputs similar to the complete test
    depth_map = None
    pose_enc = None

    # Extract direct outputs from ONNX model
    print(f"\nAggregator output shapes:")
    depth_map = None
    pose_enc = None
    world_points = None
    world_points_conf = None

    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
        if key == "depth":
            print(f"    → Found depth map in {key}")
            depth_map = value
        elif key == "pose_enc":
            print(f"    → Found pose encoding in {key}")
            pose_enc = value
        elif key == "world_points":
            print(f"    → Found world points in {key}")
            world_points = value
        elif key == "world_points_conf":
            print(f"    → Found world points confidence in {key}")
            world_points_conf = value

except Exception as e:
    print(f"Split ONNX inference failed: {e}")
    import traceback
    traceback.print_exc()
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
    print("Saving split ONNX depth images...")
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
        depth_filename = f"onnx_split_test_depth_{i:02d}.png"
        cv2.imwrite(depth_filename, depth_colored)
        print(f"Saved split ONNX depth image: {depth_filename}")

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

            # Use direct world points from ONNX model (no need to compute)
            if world_points is not None:
                # Remove batch dimension if present
                if world_points.ndim == 5:  # (1, S, H, W, 3)
                    world_points = world_points.squeeze(0)
                print(f"Direct world points shape: {world_points.shape}")

                # Also process world points confidence if available
                if world_points_conf is not None:
                    if world_points_conf.ndim == 4:  # (1, S, H, W)
                        world_points_conf = world_points_conf.squeeze(0)
                    print(f"World points confidence shape: {world_points_conf.shape}")

            # Save all predictions
            onnx_split_predictions = {
                "depth": depth_map,
                "pose_enc": pose_enc,
                "extrinsic": extrinsic_np,
                "intrinsic": intrinsic_np,
                "world_points": world_points,
                "world_points_conf": world_points_conf,
                "patch_tokens": patch_tokens
            }

        except Exception as e:
            print(f"Error in pose encoding processing: {e}")
            print("Skipping camera parameter computation, saving depth only")
            # Save predictions without camera parameters but include direct world points
            onnx_split_predictions = {
                "depth": depth_map,
                "pose_enc": pose_enc,
                "world_points": world_points,
                "world_points_conf": world_points_conf,
                "patch_tokens": patch_tokens
            }

        # Add other predictions if available
        for key, value in predictions.items():
            if key not in ["depth", "pose_enc"]:  # Don't duplicate processed data
                onnx_split_predictions[f"raw_{key}"] = value

        prediction_save_path = "onnx_split_test_predictions.npz"
        np.savez(prediction_save_path, **onnx_split_predictions)
        print(f"Saved split ONNX predictions to: {prediction_save_path}")

        # Initialize rerun for visualization
        print("Initializing rerun visualization...")
        rr.init("VGGT_ONNX_Split")
        rr.spawn()

        # Log original images
        original_images = images.cpu().numpy().squeeze(0)  # Remove batch dimension (S, C, H, W)
        for i in range(original_images.shape[0]):
            img = original_images[i].transpose(1, 2, 0)  # CHW to HWC
            # Denormalize image (assuming it was normalized to [-1, 1] or [0, 1])
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            rr.log(f"images/frame_{i:02d}", rr.Image(img))

        # Log depth images
        for i in range(depth_map.shape[0]):
            depth_normalized = depth_map[i].squeeze()
            depth_min, depth_max = depth_normalized.min(), depth_normalized.max()
            if depth_max > depth_min:
                depth_vis = (depth_normalized - depth_min) / (depth_max - depth_min)
            else:
                depth_vis = np.zeros_like(depth_normalized)
            rr.log(f"depth/frame_{i:02d}", rr.DepthImage(depth_vis))

        # Visualize pointcloud with rerun if available
        if "world_points" in onnx_split_predictions and onnx_split_predictions["world_points"] is not None:
            print("Adding colored pointcloud to rerun visualization...")

            # Get world points
            world_points_vis = onnx_split_predictions["world_points"]  # (S, H, W, 3)

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

            print(f"Visualized {len(points_valid)} colored points out of {len(points_flattened)} total points")

else:
    print("Depth predictions not found in expected format")

    # Save raw outputs anyway
    raw_save_path = "onnx_split_test_raw_outputs.npz"
    raw_predictions = predictions.copy()
    raw_predictions["patch_tokens"] = patch_tokens
    np.savez(raw_save_path, **raw_predictions)
    print(f"Saved raw split ONNX outputs to: {raw_save_path}")

print("Split ONNX model test completed successfully!")
print(f"Split ONNX depth images saved as: onnx_split_test_depth_00.png, onnx_split_test_depth_01.png, onnx_split_test_depth_02.png")

# Compare with original predictions if available
try:
    print("\nComparing with complete model predictions...")
    original_predictions = np.load("onnx_complete_test_predictions.npz")

    if "depth" in original_predictions and depth_map is not None:
        original_depth = original_predictions["depth"]

        # Compare depth maps
        depth_diff = np.abs(depth_map - original_depth)
        max_diff = np.max(depth_diff)
        mean_diff = np.mean(depth_diff)

        print(f"Depth comparison - Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-4:
            print("✓ Depth maps match closely!")
        else:
            print("⚠ Depth maps have significant differences")

    if "pose_enc" in original_predictions and pose_enc is not None:
        original_pose = original_predictions["pose_enc"]

        # Compare pose encodings
        pose_diff = np.abs(pose_enc - original_pose)
        max_diff = np.max(pose_diff)
        mean_diff = np.mean(pose_diff)

        print(f"Pose encoding comparison - Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-4:
            print("✓ Pose encodings match closely!")
        else:
            print("⚠ Pose encodings have significant differences")

except Exception as e:
    print(f"Could not compare with original predictions: {e}")
