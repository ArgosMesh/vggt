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
import rerun.blueprint as rrb

sys.path.append("vggt/")

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cpu"

def draw_pose(transform: np.ndarray, name: str, static: bool = False):
    """Draw camera pose with RGB axes arrows"""
    rr.log(name,
        rr.Arrows3D(origins=[0,0,0], vectors= [[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    radii=[0.001, 0.001, 0.001]
        ),
        static=static,
    )

    rr.log(name,
        rr.Transform3D(
            translation=transform[:3, 3],
            mat3x3=transform[:3, :3],
        ),
        static=static,
    )

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

# Load ONNX models
print("Loading split ONNX models...")
image_encoder_session = ort.InferenceSession("vggt_image_encoder.onnx")
aggregator_session = ort.InferenceSession("vggt_aggregator.onnx")

# Get model input/output info
encoder_input_info = image_encoder_session.get_inputs()[0]
encoder_output_info = image_encoder_session.get_outputs()[0]

aggregator_input_info = aggregator_session.get_inputs()
aggregator_output_info = aggregator_session.get_outputs()

# Load images
image_paths = [
    "examples/gq/01.png",
    "examples/gq/02.png",
]

print(f"Loading {len(image_paths)} images...")
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
print("\nRunning ONNX inference...")
try:
    # Process each frame separately through Image Encoder
    patch_tokens_list = []
    for i in range(S):
        single_frame = images_np[:, i]  # [B, 3, H, W]
        encoder_outputs = image_encoder_session.run(None, {encoder_input_info.name: single_frame})
        frame_tokens = encoder_outputs[0]  # [B, P, C]
        patch_tokens_list.append(frame_tokens)

    # Concatenate all frame tokens in batch dimension
    patch_tokens = np.concatenate(patch_tokens_list, axis=0)  # [B*S, P, C]

    # Run Aggregator
    aggregator_inputs = {}
    for inp in aggregator_input_info:
        if inp.name == "patch_tokens":
            aggregator_inputs["patch_tokens"] = patch_tokens
        elif inp.name == "sequence_length":
            aggregator_inputs["sequence_length"] = np.array([S], dtype=np.int64)
        elif inp.name == "height":
            aggregator_inputs["height"] = np.array([H], dtype=np.int64)
        elif inp.name == "width":
            aggregator_inputs["width"] = np.array([W], dtype=np.int64)

    aggregator_outputs = aggregator_session.run(None, aggregator_inputs)
    print("ONNX inference successful!")

    # Parse outputs
    predictions = {}
    for i, output in enumerate(aggregator_outputs):
        output_name = aggregator_output_info[i].name
        predictions[output_name] = output

    # Extract key outputs
    depth_map = predictions.get("depth")
    pose_enc = predictions.get("pose_enc")
    world_points = predictions.get("world_points")
    world_points_conf = predictions.get("world_points_conf")

except Exception as e:
    print(f"ONNX inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Process depth predictions
if depth_map is not None:
    # Remove batch dimension if present
    if depth_map.ndim == 5:  # (1, S, H, W, 1)
        depth_map = depth_map.squeeze(0)
    elif depth_map.ndim == 4 and depth_map.shape[0] == 1:  # (1, H, W, 1)
        depth_map = depth_map.squeeze(0)

    # Save depth images
    print("\nSaving depth images...")
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
        print(f"Saved: {depth_filename}")

    # Compute camera parameters from pose encoding
    if pose_enc is not None:
        print("\nComputing camera parameters...")
        
        # Handle different possible shapes
        if pose_enc.ndim == 3:
            if pose_enc.shape[0] == 1:  # (1, S, 9)
                pose_enc = pose_enc.squeeze(0)  # (S, 9)
            elif pose_enc.shape[1] == 1:  # (S, 1, 9) 
                pose_enc = pose_enc.squeeze(1)  # (S, 9)

        # Convert to torch tensor for processing
        pose_enc_tensor = torch.from_numpy(pose_enc)
        
        # Add batch dimension if needed (function expects BxSx9 or at least 3D tensor)
        if pose_enc_tensor.ndim == 2:
            pose_enc_tensor = pose_enc_tensor.unsqueeze(0)  # Add batch dimension
        
        image_shape = images.shape[-2:]

        try:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc_tensor, image_shape)

            # Convert back to numpy
            extrinsic_np = extrinsic.numpy()
            intrinsic_np = intrinsic.numpy()
            
            # Remove batch dimension if we added it
            if extrinsic_np.ndim == 4 and extrinsic_np.shape[0] == 1:
                extrinsic_np = extrinsic_np.squeeze(0)
                intrinsic_np = intrinsic_np.squeeze(0)

            print(f"Extrinsic shape: {extrinsic_np.shape}")
            print(f"Intrinsic shape: {intrinsic_np.shape}")

            # Process world points
            if world_points is not None:
                # Remove batch dimension if present
                if world_points.ndim == 5:  # (1, S, H, W, 3)
                    world_points = world_points.squeeze(0)

                if world_points_conf is not None:
                    if world_points_conf.ndim == 4:  # (1, S, H, W)
                        world_points_conf = world_points_conf.squeeze(0)

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
            print("Skipping camera parameter computation")
            # Save predictions without camera parameters
            onnx_split_predictions = {
                "depth": depth_map,
                "pose_enc": pose_enc,
                "world_points": world_points,
                "world_points_conf": world_points_conf,
                "patch_tokens": patch_tokens
            }

        # Add other predictions if available
        for key, value in predictions.items():
            if key not in ["depth", "pose_enc", "world_points", "world_points_conf"]:
                onnx_split_predictions[f"raw_{key}"] = value

        prediction_save_path = "onnx_split_test_predictions.npz"
        np.savez(prediction_save_path, **onnx_split_predictions)
        print(f"\nSaved predictions to: {prediction_save_path}")

        # Initialize rerun for visualization
        print("\nInitializing Rerun visualization...")
        rr.init("VGGT_ONNX_Split_Pointcloud")
        rr.spawn()

        # Set up blueprint with camera views
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="body/pose", contents="body/**"),
                    rrb.Vertical(
                        rrb.Spatial2DView(origin="body/cam/image"),
                        rrb.Spatial2DView(origin="body/cam/depth_map"),
                    )
                )
            )
        )

        # Get data for visualization
        original_images = images.cpu().numpy().squeeze(0)  # Remove batch dimension (S, C, H, W)
        
        # Visualize pointcloud if available
        if "world_points" in onnx_split_predictions and onnx_split_predictions["world_points"] is not None:
            # Get world points
            world_points_vis = onnx_split_predictions["world_points"]  # (S, H, W, 3)

            # Flatten the world points from all frames
            points_flattened = world_points_vis.reshape(-1, 3)

            # Extract colors from original images
            colors_array = original_images.transpose(0, 2, 3, 1)  # (S, C, H, W) -> (S, H, W, C)
            colors_flattened = colors_array.reshape(-1, 3)  # Flatten to match points

            # Denormalize colors to [0, 255] range
            colors_flattened = np.clip(colors_flattened, 0, 1)
            colors_flattened = (colors_flattened * 255).astype(np.uint8)

            # Remove invalid points
            valid_mask = np.all(np.isfinite(points_flattened), axis=1)
            valid_mask &= np.linalg.norm(points_flattened, axis=1) < 100  # Remove points too far away

            points_valid = points_flattened[valid_mask]
            colors_valid = colors_flattened[valid_mask]

            # Log the colored pointcloud
            rr.log("body/pointcloud", rr.Points3D(points_valid, colors=colors_valid, radii=0.001))

        # Log camera poses and images
        if "extrinsic" in onnx_split_predictions and "intrinsic" in onnx_split_predictions:
            extrinsic_np = onnx_split_predictions["extrinsic"]
            intrinsic_np = onnx_split_predictions["intrinsic"]
            
            for i in range(original_images.shape[0]):
                rr.set_time_sequence("frame", i)
                
                # Convert extrinsic (world-to-camera) to camera-to-world transform
                world_to_camera = np.eye(4)
                world_to_camera[:3, :4] = extrinsic_np[i]
                camera_to_world = np.linalg.inv(world_to_camera)
                
                # Log camera pinhole model
                rr.log("body/cam",
                    rr.Pinhole(
                        image_from_camera=intrinsic_np[i],
                        width=original_images.shape[3],
                        height=original_images.shape[2],
                        image_plane_distance=0.02,
                    )
                )
                
                # Log camera transform
                rr.log("body/cam", rr.Transform3D(
                    translation=camera_to_world[:3, 3],
                    mat3x3=camera_to_world[:3, :3],
                ))
                
                # Draw camera pose
                draw_pose(camera_to_world, f"body/pose{i}", static=True)
                draw_pose(camera_to_world, "body/pose")
                
                # Log original image
                img = original_images[i].transpose(1, 2, 0)  # CHW to HWC
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                rr.log("body/cam/image", rr.Image(img))

                # Log depth image
                depth_normalized = depth_map[i].squeeze()
                depth_min, depth_max = depth_normalized.min(), depth_normalized.max()
                if depth_max > depth_min:
                    depth_vis = (depth_normalized - depth_min) / (depth_max - depth_min)
                else:
                    depth_vis = np.zeros_like(depth_normalized)
                rr.log("body/cam/depth_map", rr.DepthImage(depth_vis))
                
                # Log per-frame pointcloud
                if world_points_vis is not None:
                    frame_points = world_points_vis[i].reshape(-1, 3)
                    frame_colors = original_images[i].transpose(1, 2, 0).reshape(-1, 3)
                    frame_colors = np.clip(frame_colors, 0, 1)
                    frame_colors = (frame_colors * 255).astype(np.uint8)
                    
                    # Filter invalid points for this frame
                    frame_valid_mask = np.all(np.isfinite(frame_points), axis=1)
                    frame_valid_mask &= np.linalg.norm(frame_points, axis=1) < 100
                    
                    frame_points_valid = frame_points[frame_valid_mask]
                    frame_colors_valid = frame_colors[frame_valid_mask]
                    
                    rr.log(f"body/points{i}", rr.Points3D(frame_points_valid, colors=frame_colors_valid, radii=0.0003), static=True)

            if world_points_vis is not None:
                print(f"Visualized {len(points_valid)} colored points out of {len(points_flattened)} total points")
        else:
            # If no camera parameters, still log images and depth
            print("Camera parameters not available, logging images and depth only...")
            for i in range(original_images.shape[0]):
                rr.set_time_sequence("frame", i)
                
                # Log original image
                img = original_images[i].transpose(1, 2, 0)  # CHW to HWC
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                rr.log(f"images/frame_{i:02d}", rr.Image(img))

                # Log depth image
                depth_normalized = depth_map[i].squeeze()
                depth_min, depth_max = depth_normalized.min(), depth_normalized.max()
                if depth_max > depth_min:
                    depth_vis = (depth_normalized - depth_min) / (depth_max - depth_min)
                else:
                    depth_vis = np.zeros_like(depth_normalized)
                rr.log(f"depth/frame_{i:02d}", rr.DepthImage(depth_vis))

else:
    print("Depth predictions not found in expected format")

    # Save raw outputs anyway
    raw_save_path = "onnx_split_test_raw_outputs.npz"
    raw_predictions = predictions.copy()
    raw_predictions["patch_tokens"] = patch_tokens
    np.savez(raw_save_path, **raw_predictions)
    print(f"Saved raw ONNX outputs to: {raw_save_path}")

print("\nONNX split model test completed successfully!")