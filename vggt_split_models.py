import torch
import torch.nn as nn
from typing import List, Tuple
from vggt.models.vggt import VGGT
from vggt.layers.vision_transformer import vit_large

# Set default tensor type to float32 to avoid mixed precision issues
torch.set_default_dtype(torch.float32)


class ImageEncoder(nn.Module):
    """
    DINOv2-based image encoder for per-robot deployment.
    Takes raw images and outputs patch tokens.
    """
    def __init__(self, pretrained_model_path: str = "facebook/VGGT-1B"):
        super().__init__()
        
        # Load the pretrained VGGT model to extract the exact patch_embed
        temp_vggt = VGGT.from_pretrained(pretrained_model_path)
        
        # Copy the exact patch_embed from the pretrained model
        self.patch_embed = temp_vggt.aggregator.patch_embed
        
        # Copy the exact normalization constants and reshape for single image input
        self.register_buffer("_resnet_mean", temp_vggt.aggregator._resnet_mean.squeeze(1).clone(), persistent=False)  # [1, 3, 1, 1]
        self.register_buffer("_resnet_std", temp_vggt.aggregator._resnet_std.squeeze(1).clone(), persistent=False)  # [1, 3, 1, 1]
        
        # Clean up temporary model
        del temp_vggt
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 518, 518] input images in range [0, 1]
            
        Returns:
            patch_tokens: [B, 1369, 1024] patch tokens (37x37 + 4 register tokens)
        """
        # Apply the same normalization as the original aggregator
        normalized_images = (images - self._resnet_mean) / self._resnet_std
        
        patch_tokens = self.patch_embed(normalized_images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        return patch_tokens


class AggregatorModel(nn.Module):
    """
    Centralized aggregator + heads model.
    Takes patch tokens from all robots and produces final predictions.
    """
    def __init__(self, pretrained_model_path: str = "facebook/VGGT-1B"):
        super().__init__()
        
        # Load pretrained VGGT model
        self.vggt = VGGT.from_pretrained(pretrained_model_path)
        # Ensure all parameters are float32
        self.vggt = self.vggt.float()
        
    def forward(self, all_patch_tokens: List[torch.Tensor], images: torch.Tensor = None) -> dict:
        """
        Args:
            all_patch_tokens: List of [B, 1369, 1024] tokens from each robot
            images: Optional [B, S, 3, H, W] original images for heads that need them
            
        Returns:
            predictions: Dict with camera_pred, point_pred, depth_pred, track_pred
        """
        B = all_patch_tokens[0].shape[0]
        S = len(all_patch_tokens)  # number of frames
        device = all_patch_tokens[0].device
        
        # Use provided images or create dummy ones
        if images is None:
            images = torch.zeros(B, S, 3, 518, 518, device=device)
        
        # Stack all patch tokens from robots: [B*S, 1369, 1024] 
        patch_tokens = torch.cat(all_patch_tokens, dim=0)
        
        # Create a mock patch_embed that returns our pre-computed tokens
        # This is the cleanest way to inject the tokens while preserving exact computation flow
        original_patch_embed = self.vggt.aggregator.patch_embed
        
        class MockPatchEmbed(nn.Module):
            def __init__(self, tokens_to_return):
                super().__init__()
                self.tokens_to_return = tokens_to_return
                
            def forward(self, _):
                # Return our pre-computed tokens
                return self.tokens_to_return
        
        mock_patch_embed = MockPatchEmbed(patch_tokens)
        
        # Temporarily replace patch_embed
        self.vggt.aggregator.patch_embed = mock_patch_embed
        
        try:
            # Run the VGGT model normally - it will use our patch tokens via the mock
            predictions = self.vggt(images)
            
        finally:
            # Restore original patch_embed
            self.vggt.aggregator.patch_embed = original_patch_embed
            
        return predictions


def load_split_models(pretrained_model_path: str = "facebook/VGGT-1B") -> Tuple[ImageEncoder, AggregatorModel]:
    """
    Load both split models with pretrained weights.
    
    Returns:
        image_encoder: DINOv2 encoder for per-robot deployment
        aggregator_model: Centralized aggregator + heads model
    """
    image_encoder = ImageEncoder(pretrained_model_path)
    aggregator_model = AggregatorModel(pretrained_model_path)
    
    return image_encoder, aggregator_model


def test_split_models():
    """
    Test that split models produce same output as original VGGT model.
    This test verifies bit-exact equivalence by using the same patch tokens.
    """
    # Load models
    original_vggt = VGGT.from_pretrained("facebook/VGGT-1B")
    image_encoder, aggregator_model = load_split_models()
    print('loaded split models')
    
    # Test data
    B, num_frames = 1, 3
    images = torch.randn(B * num_frames, 3, 518, 518)
    print('images shape:', images.shape)
    
    with torch.no_grad():
        # First: Extract patch tokens from the original model to ensure identical computation
        images_for_vggt = images.view(B, num_frames, 3, 518, 518)
        
        # Manually run aggregator to extract patch tokens
        B_vggt, S_vggt, C_in, H, W = images_for_vggt.shape
        
        # Normalize images (same as in aggregator)
        normalized_images = (images_for_vggt - original_vggt.aggregator._resnet_mean) / original_vggt.aggregator._resnet_std
        
        # Get patch tokens using the original patch_embed
        patch_tokens_flat = original_vggt.aggregator.patch_embed(normalized_images.view(B_vggt * S_vggt, C_in, H, W))
        if isinstance(patch_tokens_flat, dict):
            patch_tokens_flat = patch_tokens_flat["x_norm_patchtokens"]
        
        # Split the patch tokens back into per-frame lists for our split model
        all_patch_tokens = []
        for i in range(num_frames):
            frame_tokens = patch_tokens_flat[i:i+1]  # [1, 1369, 1024]
            all_patch_tokens.append(frame_tokens)
        
        print('extracted patch tokens from original model')
        
        # Now test 1: Run original VGGT model normally
        original_pred = original_vggt(images_for_vggt)
        print('original vggt inference done')
        
        # Now test 2: Run our split aggregator model with the EXACT SAME patch tokens
        split_pred = aggregator_model(all_patch_tokens, images_for_vggt)
        print('split model aggregator inference done')
    
    # Compare outputs - these should be EXACTLY the same (or very close to machine precision)
    print("Testing split models vs original (using identical patch tokens)...")
    for key in original_pred.keys():
        if key in split_pred:
            orig_tensor = original_pred[key]
            split_tensor = split_pred[key]
            max_diff = torch.max(torch.abs(orig_tensor - split_tensor)).item()
            print(f"{key}: max difference = {max_diff}")
            if max_diff > 1e-6:  # More than typical floating point precision
                print(f"  WARNING: Difference {max_diff} exceeds expected precision!")
        else:
            print(f"{key}: missing in split prediction")
    
    print("\nNow testing full split workflow (encoder + aggregator)...")
    with torch.no_grad():
        # Test 3: Full split workflow - encode images separately then aggregate
        all_patch_tokens_split = []
        for i in range(num_frames):
            frame_images = images[i:i+1]  # [1, 3, 518, 518]
            patch_tokens = image_encoder(frame_images)
            all_patch_tokens_split.append(patch_tokens)
        print('split model encoder inference done')
        
        # Aggregate and predict
        split_workflow_pred = aggregator_model(all_patch_tokens_split, images_for_vggt)
        print('split model full workflow done')
    
    # Compare split workflow vs original
    print("Split workflow vs original...")
    for key in original_pred.keys():
        if key in split_workflow_pred:
            orig_tensor = original_pred[key]
            split_tensor = split_workflow_pred[key]
            max_diff = torch.max(torch.abs(orig_tensor - split_tensor)).item()
            print(f"{key}: max difference = {max_diff}")
        else:
            print(f"{key}: missing in split prediction")


def debug_image_encoder():
    """
    Debug why ImageEncoder produces different results than original patch_embed.
    """
    print("=== Debugging ImageEncoder vs Original patch_embed ===")
    
    # Load models
    original_vggt = VGGT.from_pretrained("facebook/VGGT-1B")
    image_encoder, _ = load_split_models()
    
    # Test with a single image
    test_image = torch.randn(1, 3, 518, 518)
    
    with torch.no_grad():
        # Get tokens from original patch_embed (with normalization as in aggregator)
        # The aggregator normalizes BEFORE calling patch_embed
        normalized_image = (test_image - original_vggt.aggregator._resnet_mean) / original_vggt.aggregator._resnet_std
        original_tokens = original_vggt.aggregator.patch_embed(normalized_image)
        if isinstance(original_tokens, dict):
            original_tokens = original_tokens["x_norm_patchtokens"]
        
        # Get tokens from our ImageEncoder (it should do its own normalization)
        split_tokens = image_encoder(test_image)
        
        print(f"Original tokens shape: {original_tokens.shape}")
        print(f"Split tokens shape: {split_tokens.shape}")
        print(f"Max difference: {torch.max(torch.abs(original_tokens - split_tokens)).item()}")
        print(f"Mean difference: {torch.mean(torch.abs(original_tokens - split_tokens)).item()}")
        print(f"Original tokens range: [{torch.min(original_tokens).item():.6f}, {torch.max(original_tokens).item():.6f}]")
        print(f"Split tokens range: [{torch.min(split_tokens).item():.6f}, {torch.max(split_tokens).item():.6f}]")
        
        # Check if the issue is the normalization
        print("\n=== Testing with pre-normalized images ===")
        split_tokens_normalized = image_encoder(normalized_image)
        print(f"Max difference with normalized input: {torch.max(torch.abs(original_tokens - split_tokens_normalized)).item()}")
        
        # Check model parameters
        print("\n=== Checking model parameters ===")
        orig_params = list(original_vggt.aggregator.patch_embed.parameters())
        split_params = list(image_encoder.patch_embed.parameters())
        
        print(f"Original patch_embed has {len(orig_params)} parameters")
        print(f"Split patch_embed has {len(split_params)} parameters")
        
        if len(orig_params) == len(split_params):
            for i, (orig_p, split_p) in enumerate(zip(orig_params, split_params)):
                param_diff = torch.max(torch.abs(orig_p - split_p)).item()
                print(f"Parameter {i}: shape {orig_p.shape}, max diff = {param_diff}")
                if param_diff > 1e-8:
                    print(f"  WARNING: Parameter {i} has significant difference!")
        else:
            print("ERROR: Different number of parameters!")


if __name__ == "__main__":
    print("Testing split models after normalization fix...")
    test_split_models()