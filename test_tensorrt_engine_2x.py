# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from time import sleep
import torch
import numpy as np
import sys
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import rerun as rr

sys.path.append("vggt/")

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cpu"

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference:
    def __init__(self, engine_path):
        """Initialize TensorRT inference engine."""
        # Load engine
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            # For dynamic shapes, use max batch size or reasonable defaults
            max_shape = []
            for dim in shape:
                if dim == -1:
                    # Use reasonable defaults for dynamic dimensions
                    if tensor_name == "patch_tokens":
                        max_shape.append(2)  # Max batch size for aggregator
                    else:
                        max_shape.append(1)  # Default batch size
                else:
                    max_shape.append(dim)

            size = int(np.prod(max_shape))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append to the appropriate list
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem, 'shape': shape, 'max_shape': max_shape})
            else:
                self.outputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem, 'shape': shape, 'max_shape': max_shape})

    def infer(self, input_dict):
        """Run inference with the given inputs."""
        # Set input shapes and copy data
        for inp in self.inputs:
            if inp['name'] in input_dict:
                data = input_dict[inp['name']]
                # Set dynamic shape if needed
                if -1 in inp['shape']:
                    self.context.set_input_shape(inp['name'], data.shape)

                # Check if we need to reallocate buffer for larger input
                required_size = data.size
                current_size = inp['host'].size

                if required_size > current_size:
                    # Reallocate buffers
                    cuda.mem_free(inp['device'])
                    inp['host'] = cuda.pagelocked_empty(required_size, data.dtype)
                    inp['device'] = cuda.mem_alloc(inp['host'].nbytes)

                # Copy data to input buffer
                np.copyto(inp['host'][:data.size], data.ravel())
                cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
                # Set the tensor address for the context
                self.context.set_tensor_address(inp['name'], int(inp['device']))

        # Set output tensor addresses and reallocate if needed
        for out in self.outputs:
            # Get actual output shape from context
            output_shape = self.context.get_tensor_shape(out['name'])
            required_size = np.prod(output_shape)
            current_size = out['host'].size

            if required_size > current_size:
                # Reallocate buffers
                cuda.mem_free(out['device'])
                dtype = trt.nptype(self.engine.get_tensor_dtype(out['name']))
                out['host'] = cuda.pagelocked_empty(required_size, dtype)
                out['device'] = cuda.mem_alloc(out['host'].nbytes)

            self.context.set_tensor_address(out['name'], int(out['device']))

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs back
        output_dict = {}
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # Synchronize
        self.stream.synchronize()

        # Reshape outputs
        for out in self.outputs:
            shape = self.context.get_tensor_shape(out['name'])
            output_dict[out['name']] = out['host'][:np.prod(shape)].reshape(shape)

        return output_dict

print("Loading TensorRT engines...")
image_encoder_engine = TRTInference("vggt_image_encoder_2x.engine")
aggregator_engine = TRTInference("vggt_aggregator_2x.engine")

# Print engine information
print("="*60)
print("TENSORRT ENGINE SPECIFICATIONS:")
print("="*60)
print(f"Image Encoder TensorRT:")
print(f"  Inputs:")
for inp in image_encoder_engine.inputs:
    print(f"    {inp['name']}: shape={inp['shape']}")
print(f"  Outputs:")
for out in image_encoder_engine.outputs:
    print(f"    {out['name']}: shape={out['shape']}")

print(f"\nAggregator TensorRT:")
print(f"  Inputs:")
for inp in aggregator_engine.inputs:
    print(f"    {inp['name']}: shape={inp['shape']}")
print(f"  Outputs:")
for out in aggregator_engine.outputs:
    print(f"    {out['name']}: shape={out['shape']}")
print("="*60)

# Load first two kitchen images (same as other test scripts)
image_paths = [
    "examples/gq/01.png",
    "examples/gq/03.png",
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

# Convert to numpy for TensorRT
images_np = images.numpy()
print(images_np.shape)
# 1,2, 3, 392, 518

# Run TensorRT inference
print("\n" + "="*60)
print("ACTUAL RUNTIME DIMENSIONS:")
print("="*60)
print("Running Image Encoder TensorRT inference...")
try:
    # Process each frame separately and concatenate tokens
    patch_tokens_list = []
    for i in range(S):
        print("i=", i)
        single_frame = images_np[:, i]  # [B, 3, H, W]
        print(f"Frame {i} input shape to Image Encoder: {single_frame.shape}")
        print(f"  Frame {i} image stats - min: {single_frame.min():.3f}, max: {single_frame.max():.3f}, mean: {single_frame.mean():.3f}")

        # Debug: Check if frames are different
        if i > 0:
            frame_diff = np.abs(single_frame - images_np[:, 0]).mean()
            print(f"  Mean difference between frame {i} and frame 0: {frame_diff}")

        # Run TensorRT inference
        encoder_outputs = image_encoder_engine.infer({'images': single_frame})
        frame_tokens = encoder_outputs['patch_tokens']  # [B, P, C]
        print(f"Frame {i} output shape from Image Encoder: {frame_tokens.shape}")
        patch_tokens_list.append(frame_tokens.copy())
        sleep(2)

    # Concatenate all frame tokens in batch dimension
    patch_tokens = np.concatenate(patch_tokens_list, axis=0)  # [B*S, P, C]
    print(f"\nConcatenated patch tokens shape: {patch_tokens.shape}")

    # Debug: Check if tokens are different after concatenation
    if patch_tokens.shape[0] >= 2:
        token_diff = np.abs(patch_tokens[0] - patch_tokens[1]).mean()
        print(f"Mean difference between concatenated token 0 and 1: {token_diff}")

    print("Running Aggregator TensorRT inference...")

    # Prepare aggregator inputs
    print(f"Preparing aggregator inputs:")
    aggregator_inputs = {
        'patch_tokens': patch_tokens,
        'sequence_length': np.array([S], dtype=np.int64),
        'height': np.array([H], dtype=np.int64),
        'width': np.array([W], dtype=np.int64)
    }

    for name, value in aggregator_inputs.items():
        if isinstance(value, np.ndarray):
            print(f"  {name}: {value.shape if len(value.shape) > 0 else value}")

    # Run aggregator inference
    aggregator_outputs = aggregator_engine.infer(aggregator_inputs)
    print("TensorRT inference successful!")

    # Parse outputs
    predictions = aggregator_outputs

    print(f"\nTensorRT Output Names:")
    for key in predictions.keys():
        print(f"  {key}: {predictions[key].shape}")

    # Process outputs similar to the ONNX test
    depth_map = None
    pose_enc = None
    world_points = None
    world_points_conf = None

    for key, value in predictions.items():
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
    print(f"TensorRT inference failed: {e}")
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
    print("Saving TensorRT depth images...")
    print(f"Depth map shape for saving: {depth_map.shape}")
    for i, depth in enumerate(depth_map):
        # Normalize depth to 0-255 range for visualization
        depth_normalized = depth.squeeze()  # Remove channel dimension if present
        depth_min, depth_max = depth_normalized.min(), depth_normalized.max()

        print(f"  Frame {i} depth stats - min: {depth_min:.3f}, max: {depth_max:.3f}, mean: {depth_normalized.mean():.3f}")

        if depth_max > depth_min:
            depth_vis = ((depth_normalized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth_normalized, dtype=np.uint8)

        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)

        # Save depth image
        depth_filename = f"tensorrt_test_depth_{i:02d}.png"
        cv2.imwrite(depth_filename, depth_colored)
        print(f"Saved TensorRT depth image: {depth_filename}")

        # Also save raw depth for debugging
        np.save(f"tensorrt_raw_depth_{i:02d}.npy", depth_normalized)

    # If pose encoding is available, compute camera parameters
    if pose_enc is not None:
        print("Computing camera parameters from pose encoding...")
        print(f"Original pose_enc shape: {pose_enc.shape}")

        # Remove batch dimension if present
        if pose_enc.ndim == 3 and pose_enc.shape[0] == 1:  # Shape: (1, S, 9)
            pose_enc = pose_enc.squeeze(0)  # Shape: (S, 9)

        print(f"Processed pose_enc shape: {pose_enc.shape}")

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

            # Save all predictions
            tensorrt_predictions = {
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
            # Save predictions without camera parameters
            tensorrt_predictions = {
                "depth": depth_map,
                "pose_enc": pose_enc,
                "world_points": world_points,
                "world_points_conf": world_points_conf,
                "patch_tokens": patch_tokens
            }

        # Add other predictions if available
        for key, value in predictions.items():
            if key not in ["depth", "pose_enc", "world_points", "world_points_conf"]:
                tensorrt_predictions[f"raw_{key}"] = value

        prediction_save_path = "tensorrt_test_predictions.npz"
        np.savez(prediction_save_path, **tensorrt_predictions)
        print(f"Saved TensorRT predictions to: {prediction_save_path}")

        # Initialize rerun for visualization
        print("Initializing rerun visualization...")
        rr.init("VGGT_TensorRT")
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
        if world_points is not None:
            print("Adding colored pointcloud to rerun visualization...")

            # Get world points
            world_points_vis = world_points  # (S, H, W, 3)
            if world_points_vis.ndim == 5 and world_points_vis.shape[0] == 1:
                world_points_vis = world_points_vis.squeeze(0)

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
    raw_save_path = "tensorrt_test_raw_outputs.npz"
    raw_predictions = predictions.copy()
    raw_predictions["patch_tokens"] = patch_tokens
    np.savez(raw_save_path, **raw_predictions)
    print(f"Saved raw TensorRT outputs to: {raw_save_path}")

print("TensorRT model test completed successfully!")
print(f"TensorRT depth images saved as: tensorrt_test_depth_00.png, tensorrt_test_depth_01.png")

# Compare with ONNX predictions if available
try:
    print("\nComparing with ONNX model predictions...")
    onnx_predictions = np.load("onnx_split_test_predictions.npz")

    if "depth" in onnx_predictions and depth_map is not None:
        onnx_depth = onnx_predictions["depth"]

        # Compare depth maps
        depth_diff = np.abs(depth_map - onnx_depth)
        max_diff = np.max(depth_diff)
        mean_diff = np.mean(depth_diff)

        print(f"Depth comparison - Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-3:
            print("✓ Depth maps match closely!")
        else:
            print("⚠ Depth maps have significant differences")

    if "pose_enc" in onnx_predictions and pose_enc is not None:
        onnx_pose = onnx_predictions["pose_enc"]

        # Compare pose encodings
        pose_diff = np.abs(pose_enc - onnx_pose)
        max_diff = np.max(pose_diff)
        mean_diff = np.mean(pose_diff)

        print(f"Pose encoding comparison - Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-3:
            print("✓ Pose encodings match closely!")
        else:
            print("⚠ Pose encodings have significant differences")

    if "world_points" in onnx_predictions and world_points is not None:
        onnx_world_points = onnx_predictions["world_points"]

        # Compare world points
        world_diff = np.abs(world_points - onnx_world_points)
        max_diff = np.max(world_diff)
        mean_diff = np.mean(world_diff)

        print(f"World points comparison - Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-3:
            print("✓ World points match closely!")
        else:
            print("⚠ World points have significant differences")

except Exception as e:
    print(f"Could not compare with ONNX predictions: {e}")
