# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import cv2
import onnxruntime as ort
import sys
from PIL import Image

sys.path.append("vggt/")

from vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def test_onnx_inference(onnx_model_path="vggt.onnx", image_paths=None, output_prefix="onnx_test"):
    """
    Test ONNX inference for depth prediction from VGGT model.
    
    Args:
        onnx_model_path: Path to the ONNX model file
        image_paths: List of image paths to test with
        output_prefix: Prefix for output files
    """
    
    # Default to kitchen images if not specified
    if image_paths is None:
        image_paths = [
            "examples/kitchen/images/00.png",
            "examples/kitchen/images/01.png", 
            "examples/kitchen/images/02.png"
        ]
    
    print(f"Testing ONNX inference with model: {onnx_model_path}")
    print(f"Using {len(image_paths)} test images")
    
    # Check if ONNX model exists
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_model_path}")
    
    # Load ONNX model
    print("Loading ONNX model...")
    session = ort.InferenceSession(onnx_model_path)
    
    # Get model input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()
    
    print(f"Model input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
    print(f"Model outputs: {[out.name for out in output_info]}")
    
    # Load and preprocess images
    print(f"Loading and preprocessing {len(image_paths)} images...")
    images = load_and_preprocess_images(image_paths)
    print(f"Preprocessed images shape: {images.shape}")
    
    # Convert to numpy for ONNX
    images_np = images.numpy()
    
    # Run ONNX inference
    print("Running ONNX inference...")
    try:
        outputs = session.run(None, {input_info.name: images_np})
        print("ONNX inference successful!")
        
        # Parse outputs based on the model structure
        predictions = {}
        for i, output in enumerate(outputs):
            output_name = output_info[i].name
            predictions[output_name] = output
        
        print(f"Prediction keys: {list(predictions.keys())}")
        
        # Try to identify depth from the output shapes
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
        
        # Process depth predictions if found
        if depth_map is not None:
            print(f"Processing depth map with shape: {depth_map.shape}")
            
            # Remove batch dimension if present
            if depth_map.ndim == 5:  # (1, S, H, W, 1)
                depth_map = depth_map.squeeze(0)
            elif depth_map.ndim == 4 and depth_map.shape[0] == 1:  # (1, H, W, 1)
                depth_map = depth_map.squeeze(0)
            
            # Save depth images
            print("Saving ONNX depth predictions...")
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
                depth_filename = f"{output_prefix}_depth_{i:02d}.png"
                cv2.imwrite(depth_filename, depth_colored)
                print(f"Saved ONNX depth image: {depth_filename}")
            
            # If pose encoding is available, compute camera parameters
            if pose_enc is not None:
                print("Computing camera parameters from pose encoding...")
                if pose_enc.ndim == 3 and pose_enc.shape[0] == 1:  # Remove batch dimension
                    pose_enc = pose_enc.squeeze(0)
                
                # Convert to torch tensor for processing (the utility functions expect torch tensors)
                import torch
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
                
                prediction_save_path = f"{output_prefix}_predictions.npz"
                np.savez(prediction_save_path, **onnx_predictions)
                print(f"Saved ONNX predictions to: {prediction_save_path}")
        
        else:
            print("Depth predictions not found in expected format")
            
            # Save raw outputs anyway
            raw_save_path = f"{output_prefix}_raw_outputs.npz"
            np.savez(raw_save_path, **predictions)
            print(f"Saved raw ONNX outputs to: {raw_save_path}")
        
        return predictions
        
    except Exception as e:
        print(f"ONNX inference failed: {e}")
        raise

def test_split_onnx_inference(
    image_encoder_path="vggt_image_encoder.onnx", 
    aggregator_path="vggt_aggregator.onnx",
    image_paths=None, 
    output_prefix="split_onnx_test"
):
    """
    Test ONNX inference using split models (image encoder + aggregator).
    
    Args:
        image_encoder_path: Path to the image encoder ONNX model
        aggregator_path: Path to the aggregator ONNX model  
        image_paths: List of image paths to test with
        output_prefix: Prefix for output files
    """
    
    # Default to kitchen images if not specified
    if image_paths is None:
        image_paths = [
            "examples/kitchen/images/00.png",
            "examples/kitchen/images/01.png", 
            "examples/kitchen/images/02.png"
        ]
    
    print(f"Testing split ONNX inference:")
    print(f"  Image encoder: {image_encoder_path}")
    print(f"  Aggregator: {aggregator_path}")
    print(f"  Using {len(image_paths)} test images")
    
    # Check if ONNX models exist
    if not os.path.exists(image_encoder_path):
        raise FileNotFoundError(f"Image encoder ONNX model not found at: {image_encoder_path}")
    if not os.path.exists(aggregator_path):
        raise FileNotFoundError(f"Aggregator ONNX model not found at: {aggregator_path}")
    
    # Load ONNX models
    print("Loading ONNX models...")
    encoder_session = ort.InferenceSession(image_encoder_path)
    aggregator_session = ort.InferenceSession(aggregator_path)
    
    # Get model input/output info
    encoder_input_info = encoder_session.get_inputs()[0]
    encoder_output_info = encoder_session.get_outputs()[0]
    aggregator_input_info = aggregator_session.get_inputs()[0]
    aggregator_output_info = aggregator_session.get_outputs()
    
    print(f"Encoder input: {encoder_input_info.name}, shape: {encoder_input_info.shape}")
    print(f"Encoder output: {encoder_output_info.name}, shape: {encoder_output_info.shape}")
    print(f"Aggregator input: {aggregator_input_info.name}, shape: {aggregator_input_info.shape}")
    print(f"Aggregator outputs: {[out.name for out in aggregator_output_info]}")
    
    # Load and preprocess images for split models (need square 518x518)
    print(f"Loading and preprocessing {len(image_paths)} images for split models...")
    images, _ = load_and_preprocess_images_square(image_paths, target_size=518)
    print(f"Preprocessed images shape: {images.shape}")
    
    # Run image encoder for each image
    print("Running image encoder...")
    all_tokens = []
    for i, single_image in enumerate(images):
        # Add batch dimension and convert to numpy
        single_image_batch = single_image.unsqueeze(0).numpy()
        
        # Run encoder
        tokens = encoder_session.run(None, {encoder_input_info.name: single_image_batch})[0]
        all_tokens.append(tokens)
        print(f"Encoded image {i}: tokens shape {tokens.shape}")
    
    # Concatenate tokens for aggregator
    concat_tokens = np.concatenate(all_tokens, axis=1)  # Concatenate along token dimension
    print(f"Concatenated tokens shape: {concat_tokens.shape}")
    
    # Run aggregator
    print("Running aggregator...")
    try:
        aggregator_outputs = aggregator_session.run(None, {aggregator_input_info.name: concat_tokens})
        print("Split ONNX inference successful!")
        
        # Parse aggregator outputs
        output_names = ['pose_enc', 'world_points', 'depth', 'track']
        predictions = {}
        for i, output in enumerate(aggregator_outputs):
            if i < len(output_names):
                predictions[output_names[i]] = output
            else:
                predictions[f"output_{i}"] = output
        
        print(f"Prediction keys: {list(predictions.keys())}")
        for key, value in predictions.items():
            print(f"{key}: shape={value.shape}")
        
        # Process depth predictions
        if "depth" in predictions:
            depth_map = predictions["depth"]
            print(f"Processing depth map with shape: {depth_map.shape}")
            
            # Remove batch dimension if present
            if depth_map.ndim == 5:  # (1, S, H, W, 1)
                depth_map = depth_map.squeeze(0)
            elif depth_map.ndim == 4 and depth_map.shape[0] == 1:  # (1, H, W, 1)
                depth_map = depth_map.squeeze(0)
            
            # Save depth images
            print("Saving split ONNX depth predictions...")
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
                depth_filename = f"{output_prefix}_depth_{i:02d}.png"
                cv2.imwrite(depth_filename, depth_colored)
                print(f"Saved split ONNX depth image: {depth_filename}")
            
            # Process pose encoding and compute camera parameters
            if "pose_enc" in predictions:
                print("Computing camera parameters from pose encoding...")
                pose_enc = predictions["pose_enc"]
                print(f"Original pose_enc shape: {pose_enc.shape}")
                if pose_enc.ndim == 3 and pose_enc.shape[0] == 1:  # Remove batch dimension
                    pose_enc = pose_enc.squeeze(0)
                print(f"After squeeze pose_enc shape: {pose_enc.shape}")
                
                # Convert to torch tensor for processing
                import torch
                pose_enc_tensor = torch.from_numpy(pose_enc)
                image_shape = images.shape[-2:]
                print(f"Image shape for pose processing: {image_shape}")
                
                try:
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc_tensor, image_shape)
                except Exception as e:
                    print(f"Error in pose encoding processing: {e}")
                    print(f"pose_enc_tensor shape: {pose_enc_tensor.shape}")
                    # Skip pose processing but continue with depth
                    extrinsic = None
                    intrinsic = None
                
                if extrinsic is not None and intrinsic is not None:
                    # Convert back to numpy
                    extrinsic_np = extrinsic.numpy()
                    intrinsic_np = intrinsic.numpy()
                    
                    print(f"Extrinsic matrices shape: {extrinsic_np.shape}")
                    print(f"Intrinsic matrices shape: {intrinsic_np.shape}")
                    
                    # Generate world points from depth map
                    print("Computing world points from split ONNX depth predictions...")
                    world_points = unproject_depth_map_to_point_map(
                        torch.from_numpy(depth_map), 
                        torch.from_numpy(extrinsic_np), 
                        torch.from_numpy(intrinsic_np)
                    ).numpy()
                    
                    print(f"World points shape: {world_points.shape}")
                    
                    # Save all predictions
                    split_onnx_predictions = {
                        "depth": depth_map,
                        "pose_enc": pose_enc,
                        "extrinsic": extrinsic_np,
                        "intrinsic": intrinsic_np,
                        "world_points_from_depth": world_points
                    }
                else:
                    print("Skipping camera parameter computation due to pose encoding error")
                    # Save predictions without camera parameters
                    split_onnx_predictions = {
                        "depth": depth_map,
                        "pose_enc": pose_enc
                    }
                
                # Add other predictions
                for key, value in predictions.items():
                    if key not in ["depth", "pose_enc"]:
                        split_onnx_predictions[key] = value
                
                prediction_save_path = f"{output_prefix}_predictions.npz"
                np.savez(prediction_save_path, **split_onnx_predictions)
                print(f"Saved split ONNX predictions to: {prediction_save_path}")
        
        return predictions
        
    except Exception as e:
        print(f"Split ONNX inference failed: {e}")
        raise

def compare_pytorch_onnx_outputs(pytorch_predictions_path="predictions.npz", 
                                onnx_predictions_path="onnx_test_predictions.npz"):
    """
    Compare outputs from PyTorch and ONNX models.
    """
    print("\nComparing PyTorch vs ONNX outputs...")
    
    if not os.path.exists(pytorch_predictions_path):
        print(f"PyTorch predictions not found at: {pytorch_predictions_path}")
        return
    
    if not os.path.exists(onnx_predictions_path):
        print(f"ONNX predictions not found at: {onnx_predictions_path}")
        return
    
    pytorch_data = np.load(pytorch_predictions_path)
    onnx_data = np.load(onnx_predictions_path)
    
    print(f"PyTorch keys: {list(pytorch_data.keys())}")
    print(f"ONNX keys: {list(onnx_data.keys())}")
    
    # Compare common keys
    common_keys = set(pytorch_data.keys()) & set(onnx_data.keys())
    print(f"Common keys: {common_keys}")
    
    for key in common_keys:
        pytorch_val = pytorch_data[key]
        onnx_val = onnx_data[key]
        
        print(f"\n{key}:")
        print(f"  PyTorch shape: {pytorch_val.shape}")
        print(f"  ONNX shape: {onnx_val.shape}")
        
        if pytorch_val.shape == onnx_val.shape:
            # Compute differences
            abs_diff = np.abs(pytorch_val - onnx_val)
            rel_diff = abs_diff / (np.abs(pytorch_val) + 1e-8)
            
            print(f"  Max absolute difference: {abs_diff.max():.6f}")
            print(f"  Mean absolute difference: {abs_diff.mean():.6f}")
            print(f"  Max relative difference: {rel_diff.max():.6f}")
            print(f"  Mean relative difference: {rel_diff.mean():.6f}")
            
            # Check if they're close
            if np.allclose(pytorch_val, onnx_val, rtol=1e-3, atol=1e-5):
                print(f"  ✅ Values are close (rtol=1e-3, atol=1e-5)")
            else:
                print(f"  ❌ Values differ significantly")
        else:
            print(f"  ❌ Shape mismatch!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ONNX inference for VGGT depth prediction")
    parser.add_argument("--onnx_model", default="vggt.onnx", help="Path to monolithic ONNX model")
    parser.add_argument("--split", action="store_true", help="Use split models (image encoder + aggregator)")
    parser.add_argument("--image_encoder", default="vggt_image_encoder.onnx", help="Path to image encoder ONNX model")
    parser.add_argument("--aggregator", default="vggt_aggregator.onnx", help="Path to aggregator ONNX model")
    parser.add_argument("--image_folder", help="Folder containing test images")
    parser.add_argument("--images", nargs="+", help="Specific image paths to test")
    parser.add_argument("--output_prefix", default="onnx_test", help="Prefix for output files")
    parser.add_argument("--compare", action="store_true", help="Compare with PyTorch outputs")
    
    args = parser.parse_args()
    
    # Determine image paths
    image_paths = None
    if args.images:
        image_paths = args.images
    elif args.image_folder:
        # Get all image files from folder
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend([
                os.path.join(args.image_folder, f) 
                for f in os.listdir(args.image_folder) 
                if f.lower().endswith(ext)
            ])
        image_paths.sort()
        print(f"Found {len(image_paths)} images in {args.image_folder}")
    
    # Run ONNX inference test
    try:
        if args.split:
            # Use split models
            predictions = test_split_onnx_inference(
                image_encoder_path=args.image_encoder,
                aggregator_path=args.aggregator,
                image_paths=image_paths,
                output_prefix=args.output_prefix
            )
            print("\n✅ Split ONNX inference test completed successfully!")
        else:
            # Use monolithic model
            predictions = test_onnx_inference(
                onnx_model_path=args.onnx_model,
                image_paths=image_paths,
                output_prefix=args.output_prefix
            )
            print("\n✅ ONNX inference test completed successfully!")
        
        # Compare with PyTorch if requested
        if args.compare:
            compare_pytorch_onnx_outputs(
                onnx_predictions_path=f"{args.output_prefix}_predictions.npz"
            )
            
    except Exception as e:
        print(f"\n❌ ONNX inference test failed: {e}")
        sys.exit(1)