# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import cv2

sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cpu"

print("Initializing and loading VGGT model...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)

# Load first three kitchen images
image_paths = [
    # "examples/room/images/no_overlap_4.jpg",
    # "examples/room/images/no_overlap_2.jpg",
    # "examples/room/images/no_overlap_3.jpg",
    "examples/gq/01.png",
    "examples/gq/02.png",
    #"examples/gq/03.png",
    # "examples/gq/04.png",
    # "examples/gq/05.png",
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

# Define the Image Encoder (DINOV2 patch embedding only)
class ImageEncoder(nn.Module):
    def __init__(self, aggregator):
        super().__init__()
        self.patch_embed = aggregator.patch_embed
        # Copy normalization constants
        self.register_buffer("_resnet_mean", aggregator._resnet_mean.clone())
        self.register_buffer("_resnet_std", aggregator._resnet_std.clone())

    def forward(self, images):
        """
        Args:
            images (torch.Tensor): Input images with shape [B, 3, H, W], in range [0, 1].

        Returns:
            torch.Tensor: Patch tokens with shape [B, P, C]
        """
        B, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images using ResNet statistics
        images = (images - self._resnet_mean.squeeze(1)) / self._resnet_std.squeeze(1)

        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        return patch_tokens

# Define the Aggregator with decoder heads
class AggregatorWithHeads(nn.Module):
    def __init__(self, vggt_model):
        super().__init__()
        # Copy aggregator components except patch_embed
        self.rope = vggt_model.aggregator.rope
        self.position_getter = vggt_model.aggregator.position_getter
        self.frame_blocks = vggt_model.aggregator.frame_blocks
        self.global_blocks = vggt_model.aggregator.global_blocks
        self.depth = vggt_model.aggregator.depth
        self.aa_order = vggt_model.aggregator.aa_order
        self.patch_size = vggt_model.aggregator.patch_size
        self.aa_block_size = vggt_model.aggregator.aa_block_size
        self.aa_block_num = vggt_model.aggregator.aa_block_num
        self.camera_token = vggt_model.aggregator.camera_token
        self.register_token = vggt_model.aggregator.register_token
        self.patch_start_idx = vggt_model.aggregator.patch_start_idx

        # Copy decoder heads
        self.camera_head = vggt_model.camera_head
        self.point_head = vggt_model.point_head
        self.depth_head = vggt_model.depth_head
        self.track_head = vggt_model.track_head

    def forward(self, patch_tokens, sequence_length, height, width):
        """
        Args:
            patch_tokens (torch.Tensor): Concatenated patch tokens with shape [B*S, P, C]
            sequence_length (int): Number of frames S
            height (int): Original image height H
            width (int): Original image width W

        Returns:
            dict: Model predictions
        """
        BS, P, C = patch_tokens.shape
        B = BS // sequence_length
        S, H, W = sequence_length, height, width

        # Import the helper function
        from vggt.models.aggregator import slice_expand_and_flatten

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=patch_tokens.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(patch_tokens.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates

        # Reconstruct images tensor for decoder heads (they expect original input format)
        images = torch.zeros(B, S, 3, H, W, device=patch_tokens.device)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(output_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    output_list, images=images, patch_start_idx=self.patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    output_list, images=images, patch_start_idx=self.patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        return predictions

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

# Create the split models
print("Creating split models...")
image_encoder = ImageEncoder(model.aggregator)
aggregator_with_heads = AggregatorWithHeads(model)

# Test the split on the loaded images
print("="*60)
print("PYTORCH MODEL TESTING:")
print("="*60)
print("Testing split inference...")
S, C_in, H, W = images.shape
B = 1  # Single batch for demo

# Add batch dimension if needed
if len(images.shape) == 4:
    images = images.unsqueeze(0)  # [1, S, 3, H, W]

B, S, C_in, H, W = images.shape
print(f"Input images shape: {images.shape}")

with torch.no_grad():
    # Process each frame separately and concatenate tokens
    patch_tokens_list = []
    for i in range(S):
        single_frame = images[:, i]  # [B, 3, H, W]
        print(f"Frame {i} input shape to PyTorch Image Encoder: {single_frame.shape}")
        frame_tokens = image_encoder(single_frame)  # [B, P, C]
        print(f"Frame {i} output shape from PyTorch Image Encoder: {frame_tokens.shape}")
        patch_tokens_list.append(frame_tokens)

    # Concatenate all frame tokens in batch dimension
    patch_tokens = torch.cat(patch_tokens_list, dim=0)  # [B*S, P, C]
    print(f"\nConcatenated patch tokens shape: {patch_tokens.shape}")

    # Test aggregator with heads
    print(f"PyTorch Aggregator inputs: patch_tokens={patch_tokens.shape}, S={S}, H={H}, W={W}")
    predictions = aggregator_with_heads(patch_tokens, S, H, W)
    print(f"\nPyTorch Aggregator predictions keys: {list(predictions.keys())}")

    print(f"PyTorch Aggregator output shapes:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    print("="*60)

# Convert pose encoding to extrinsic and intrinsic matrices
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
print("Saving split model depth images...")
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
    depth_filename = f"split_test_depth_{i:02d}.png"
    cv2.imwrite(depth_filename, depth_colored)
    print(f"Saved split depth image: {depth_filename}")

# Save split model predictions
prediction_save_path = "split_test_predictions.npz"
np.savez(prediction_save_path, **predictions)
print(f"Saved split model predictions to: {prediction_save_path}")

# Export Image Encoder to ONNX (single frame input)
print("\n" + "="*60)
print("ONNX EXPORT:")
print("="*60)
print("Exporting Image Encoder to ONNX...")
single_frame_input = images[:, 0]  # [B, 3, H, W]
print(f"Image Encoder ONNX export input shape: {single_frame_input.shape}")

# Get sample output for verification
with torch.no_grad():
    sample_output = image_encoder(single_frame_input)
    print(f"Image Encoder ONNX export output shape: {sample_output.shape}")

torch.onnx.export(
    image_encoder,
    single_frame_input,
    "vggt_image_encoder.onnx",
    verbose=True,
    dynamo=True,
    report=True,
    opset_version=17,
    input_names=["images"],
    output_names=["patch_tokens"]
)

# Export Aggregator with Heads to ONNX
print("\nExporting Aggregator with Heads to ONNX...")
print(f"Aggregator ONNX export inputs:")
print(f"  patch_tokens: {patch_tokens.shape}")
print(f"  sequence_length: {S}")
print(f"  height: {H}")
print(f"  width: {W}")

# Get sample output for verification
with torch.no_grad():
    sample_predictions = aggregator_with_heads(patch_tokens, S, H, W)
    print(f"Aggregator ONNX export output shapes:")
    for key, value in sample_predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

torch.onnx.export(
    aggregator_with_heads,
    (patch_tokens, S, H, W),
    "vggt_aggregator.onnx",
    verbose=True,
    dynamo=True,
    report=True,
    opset_version=17,
    input_names=["patch_tokens", "sequence_length", "height", "width"],
    output_names=["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"]
)
print("="*60)

print("Split ONNX export completed successfully!")
print(f"Image encoder exported to: vggt_image_encoder.onnx")
print(f"Aggregator with heads exported to: vggt_aggregator.onnx")
print(f"Split model depth images saved as: split_test_depth_00.png, split_test_depth_01.png, split_test_depth_02.png")
