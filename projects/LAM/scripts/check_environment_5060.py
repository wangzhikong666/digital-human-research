"""Check LAM imports and real CUDA kernels without downloading model assets."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch

    assert torch.cuda.is_available(), "CUDA is unavailable in this process."
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}", flush=True)

    for name in (
        "torchvision", "transformers", "diffusers", "accelerate", "gradio",
        "tensorflow", "chumpy", "pytorch3d", "nvdiffrast.torch",
        "diff_gaussian_rasterization", "simple_knn._C", "app_lam",
        "lam.models.modeling_lam",
    ):
        importlib.import_module(name)
        print(f"Import OK: {name}", flush=True)

    sample = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    gpu_result = sample.cuda() @ sample.cuda().T
    torch.testing.assert_close(gpu_result.cpu(), sample @ sample.T)
    print("CUDA matrix multiplication OK", flush=True)

    from xformers.ops import memory_efficient_attention

    query = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float16)
    attention = memory_efficient_attention(query, query, query)
    assert attention.shape == query.shape and torch.isfinite(attention).all()
    print("xformers attention CUDA kernel OK", flush=True)

    from lam.models.encoders.dinov2.layers.attention import Attention, MemEffAttention

    image_attention = MemEffAttention(dim=64, num_heads=1).cuda().eval()
    reference_attention = Attention(dim=64, num_heads=1).cuda().eval()
    reference_attention.load_state_dict(image_attention.state_dict())
    image_input = torch.randn(1, 32, 64, device="cuda")
    with torch.no_grad():
        image_features = image_attention(image_input)
        reference_features = reference_attention(image_input)
    assert torch.isfinite(image_features).all()
    torch.testing.assert_close(image_features, reference_features, rtol=1e-4, atol=1e-5)
    print("LAM DINOv2 float32 attention OK", flush=True)

    from lam.models.transformer import TransformerDecoder

    decoder = TransformerDecoder(
        block_type="sd3_cond", num_layers=2, num_heads=1,
        inner_dim=64, cond_dim=64,
    ).cuda().eval()
    with torch.no_grad():
        decoded = decoder(
            torch.randn(1, 8, 64, device="cuda"),
            cond=torch.randn(1, 4, 64, device="cuda"),
        )
    assert decoded.shape == (1, 8, 64) and torch.isfinite(decoded).all()
    print("LAM transformer / diffusers API compatibility OK", flush=True)

    from pytorch3d.ops import knn_points
    from pytorch3d.renderer import rasterize_meshes
    from pytorch3d.structures import Meshes

    points = torch.randn(1, 8, 3, device="cuda")
    assert torch.isfinite(knn_points(points, points, K=2).dists).all()
    vertices = torch.tensor(
        [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]],
        device="cuda",
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda")
    mesh = Meshes(verts=[vertices], faces=[faces])
    pixels, _, _, _ = rasterize_meshes(mesh, image_size=16, faces_per_pixel=1)
    assert (pixels >= 0).any()
    print("PyTorch3D KNN and mesh rasterization CUDA kernels OK", flush=True)

    from simple_knn._C import distCUDA2

    distances = distCUDA2(points[0].contiguous())
    assert distances.shape == (8,) and torch.isfinite(distances).all()
    print("simple-knn CUDA kernel OK", flush=True)

    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer,
    )

    settings = GaussianRasterizationSettings(
        image_height=16, image_width=16, tanfovx=1.0, tanfovy=1.0,
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=torch.eye(4, device="cuda"), sh_degree=0,
        campos=torch.zeros(3, device="cuda"), prefiltered=False, debug=False,
    )
    result = GaussianRasterizer(settings)(
        means3D=torch.tensor([[0.0, 0.0, 2.0]], device="cuda"),
        means2D=torch.zeros(1, 3, device="cuda"),
        opacities=torch.full((1, 1), 0.8, device="cuda"),
        colors_precomp=torch.ones(1, 3, device="cuda"),
        scales=torch.full((1, 3), 0.1, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
    )
    assert result[0].shape == (3, 16, 16)
    assert torch.isfinite(result[0]).all() and result[0].max() > 0
    print("Gaussian rasterization CUDA kernel OK", flush=True)

    from external.landmark_detection.FaceBoxesV2.utils.nms.cpu_nms import cpu_nms

    boxes = np.array([[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8]], dtype=np.float32)
    assert len(cpu_nms(boxes, 0.5)) == 1
    print("FaceBoxes Cython NMS extension OK", flush=True)

    import nvdiffrast.torch as dr

    positions = torch.cat([vertices, torch.ones(3, 1, device="cuda")], dim=1)[None]
    context = dr.RasterizeCudaContext()
    raster, _ = dr.rasterize(context, positions, faces.int(), resolution=[16, 16])
    assert (raster[..., 3] > 0).any()
    torch.cuda.synchronize()
    print("nvdiffrast JIT compilation and CUDA rasterization OK", flush=True)
    print("LAM environment checks passed. Model assets are checked separately.", flush=True)


if __name__ == "__main__":
    main()
