import torch
import torch.onnx
import os
from vggt_split_models import ImageEncoder, AggregatorModel, load_split_models

# Set default tensor type to float32 to avoid mixed precision issues
torch.set_default_dtype(torch.float32)


def export_image_encoder(model: ImageEncoder, output_path: str = "image_encoder.onnx"):
    """
    Export ImageEncoder to ONNX format.
    """
    model.eval()
    
    # Ensure model is in float32
    model = model.float()
    
    # Create dummy input in float32
    dummy_input = torch.randn(3, 3, 518, 518, dtype=torch.float32)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['patch_tokens'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'patch_tokens': {0: 'batch_size'}
        },
        verbose=False,
        dynamo=True,  # Disable dynamo to avoid type issues
    )
    
    print(f"ImageEncoder exported to {output_path}")
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")


def export_aggregator_model(model: AggregatorModel, output_path: str = "aggregator_model.onnx"):
    """
    Export AggregatorModel to ONNX format.
    """
    model.eval()
    
    # Ensure model is in float32
    model = model.float()
    
    # Create dummy inputs - tokens from 3 robots in float32
    num_robots = 3
    dummy_tokens = [torch.randn(3, 1369, 1024, dtype=torch.float32) for _ in range(num_robots)]
    
    # ONNX doesn't handle list inputs well, so we'll concatenate and track frame boundaries
    concat_tokens = torch.cat(dummy_tokens, dim=1)  # [1, 3*1369, 1024]
    
    # Create a wrapper model that takes concatenated tokens
    class AggregatorWrapper(torch.nn.Module):
        def __init__(self, aggregator_model, num_frames):
            super().__init__()
            self.aggregator_model = aggregator_model
            self.num_frames = num_frames
            self.tokens_per_frame = 1369  # 37x37 patches + 4 register tokens
            
        def forward(self, concat_tokens):
            # Ensure input is float32
            concat_tokens = concat_tokens.float()
            
            # Split concatenated tokens back into list
            token_list = []
            for i in range(self.num_frames):
                start_idx = i * self.tokens_per_frame
                end_idx = start_idx + self.tokens_per_frame
                token_list.append(concat_tokens[:, start_idx:end_idx])
            
            # Run aggregator model
            predictions = self.aggregator_model(token_list)
            
            # Ensure all outputs are float32
            return (
                predictions['pose_enc'].float(),
                predictions['world_points'].float(), 
                predictions['depth'].float(),
                predictions.get('track', torch.zeros(1, 3, 1, 2, dtype=torch.float32)).float()
            )
    
    wrapper_model = AggregatorWrapper(model, num_robots)
    wrapper_model = wrapper_model.float()  # Ensure wrapper is also float32
    
    # # Try to script the model first to detect type issues
    # try:
    #     print("Attempting to script the wrapper model...")
    #     scripted_model = torch.jit.script(wrapper_model)
    #     print("Model scripted successfully, using scripted version for export")
    #     export_model = scripted_model
    # except Exception as e:
    #     print(f"Scripting failed: {e}, using original model")
    #     export_model = wrapper_model

    export_model = wrapper_model
    
    # Export to ONNX
    torch.onnx.export(
        export_model,
        concat_tokens,
        output_path,
        export_params=True,
        opset_version=11,  # Use older opset version for better compatibility
        do_constant_folding=True,
        input_names=['concat_tokens'],
        output_names=['pose_enc', 'world_points', 'depth', 'track'],
        dynamic_axes={
            'concat_tokens': {0: 'batch_size'},
            'pose_enc': {0: 'batch_size'},
            'world_points': {0: 'batch_size'},
            'depth': {0: 'batch_size'},
            'track': {0: 'batch_size'}
        },
        verbose=False,
        dynamo=True,
        #training=torch.onnx.TrainingMode.EVAL,  # Explicitly set to eval mode
        #keep_initializers_as_inputs=False
    )
    
    print(f"AggregatorModel exported to {output_path}")
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")


def export_both_models(
    image_encoder_path: str = "vggt_image_encoder.onnx",
    aggregator_path: str = "vggt_aggregator.onnx"
):
    """
    Export both split models to ONNX format.
    """
    print("Loading split models...")
    image_encoder, aggregator_model = load_split_models()
    
    print("\nExporting ImageEncoder...")
    export_image_encoder(image_encoder, image_encoder_path)
    
    print("\nExporting AggregatorModel...")
    export_aggregator_model(aggregator_model, aggregator_path)
    
    print(f"\nBoth models exported successfully!")
    print(f"Per-robot model: {image_encoder_path}")
    print(f"Central model: {aggregator_path}")


def test_onnx_models():
    """
    Test that ONNX models produce same output as PyTorch models.
    """
    import onnxruntime as ort
    import numpy as np
    print("Testing ONNX models...")
    # Load PyTorch models
    image_encoder, aggregator_model = load_split_models()
    
    # Test ImageEncoder
    print("Testing ImageEncoder ONNX...")
    ort_session_encoder = ort.InferenceSession("vggt_image_encoder.onnx")
    
    dummy_images = torch.randn(1, 3, 518, 518, dtype=torch.float32)
    
    # PyTorch prediction
    with torch.no_grad():
        pytorch_tokens = image_encoder(dummy_images)
    
    # ONNX prediction
    onnx_tokens = ort_session_encoder.run(
        None, 
        {"images": dummy_images.numpy().astype('float32')}
    )[0]
    
    max_diff = np.max(np.abs(pytorch_tokens.numpy() - onnx_tokens))
    print(f"ImageEncoder max difference: {max_diff}")
    
    # Test AggregatorModel
    print("Testing AggregatorModel ONNX...")
    ort_session_aggregator = ort.InferenceSession("vggt_aggregator.onnx")
    
    # Create test tokens
    num_robots = 3
    dummy_token_list = [torch.randn(1, 1369, 1024, dtype=torch.float32) for _ in range(num_robots)]
    concat_tokens = torch.cat(dummy_token_list, dim=1)
    
    # PyTorch prediction
    with torch.no_grad():
        pytorch_pred = aggregator_model(dummy_token_list)
    
    # ONNX prediction
    onnx_pred = ort_session_aggregator.run(
        None,
        {"concat_tokens": concat_tokens.numpy().astype('float32')}
    )
    
    # Compare each output
    pred_keys = ['pose_enc', 'world_points', 'depth', 'track']
    for i, key in enumerate(pred_keys):
        if key in pytorch_pred:
            max_diff = np.max(np.abs(pytorch_pred[key].numpy() - onnx_pred[i]))
            print(f"{key} max difference: {max_diff}")
        else:
            print(f"Warning: {key} not found in PyTorch predictions")


if __name__ == "__main__":
    # Export both models
    export_both_models()
    
    # Test ONNX models (requires onnxruntime)
    try:
        test_onnx_models()
    except ImportError:
        print("\nonnxruntime not installed, skipping ONNX testing")
        print("Install with: pip install onnxruntime")