import onnxruntime as ort
import torch


def run_vggt_onnx(model_path, images, query_points=None):
    """
    Run VGGT ONNX model inference

    Args:
        model_path (str): Path to the ONNX model file
        images (torch.Tensor): Input images [S, 3, H, W] or [B, S, 3, H, W]
        query_points (torch.Tensor, optional): Query points [N, 2] or [B, N, 2]

    Returns:
        dict: Model predictions
    """
    # Create inference session
    session = ort.InferenceSession(model_path)

    # Get input/output info
    output_names = [output.name for output in session.get_outputs()]

    # Prepare inputs
    if isinstance(images, torch.Tensor):
        images_np = images.cpu().numpy()
    else:
        images_np = images

    inputs = {"images": images_np}

    if query_points is not None:
        if isinstance(query_points, torch.Tensor):
            query_points_np = query_points.cpu().numpy()
        else:
            query_points_np = query_points
        inputs["query_points"] = query_points_np

    # Run inference
    outputs = session.run(output_names, inputs)

    # Create results dictionary
    results = {}
    for i, name in enumerate(output_names):
        results[name] = outputs[i]

    return results


# Usage example
# Load your data
images = torch.randn(1, 4, 3, 224, 224)  # Example: batch=1, sequence=5, RGB, 518x518
query_points = torch.randn(1, 10, 2)  # Example: batch=1, 10 query points

# Run inference
predictions = run_vggt_onnx("vggt.onnx", images, query_points)

# Access results
if "pose_enc" in predictions:
    pose_encoding = predictions["pose_enc"]
if "depth" in predictions:
    depth_maps = predictions["depth"]
if "track" in predictions:
    point_tracks = predictions["track"]
