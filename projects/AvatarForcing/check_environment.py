"""Check the local runtime without downloading or loading model checkpoints."""

import importlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Activate the Python 3.10 avatarforcing environment.")

    for module in (
        "inference",
        "preprocess_user_video",
        "models.avatarforcing.AvatarForcing",
    ):
        importlib.import_module(module)
        print(f"Import OK: {module}", flush=True)

    for package in (
        "torch", "torchvision", "torchaudio", "numpy", "transformers",
        "opencv-python", "opencv-python-headless", "face-alignment", "av",
    ):
        print(f"{package}: {importlib.metadata.version(package)}", flush=True)

    import av
    import librosa

    audio, sample_rate = librosa.load(str(root / "data/avatar.wav"), sr=16000, duration=0.1)
    if sample_rate != 16000 or audio.size == 0:
        raise RuntimeError("Audio decoding/resampling failed.")
    with av.open(str(root / "data/user.mp4")) as video:
        frame = next(video.decode(video=0)).to_ndarray(format="rgb24")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError("Video decoding failed.")
    print("Sample audio decoding/resampling and video decoding: OK", flush=True)

    import torch
    import torch.nn.functional as F
    from torchvision.ops import nms
    from models.avatarforcing.styledecoder import upfirdn2d, make_kernel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run with access to the NVIDIA GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA: {torch.version.cuda}; capability: {torch.cuda.get_device_capability(0)}")
    with torch.inference_mode():
        x = torch.randn(128, 128, device="cuda")
        torch.testing.assert_close(x @ torch.eye(128, device="cuda"), x)
        image = torch.randn(1, 3, 16, 16, device="cuda")
        result = upfirdn2d(image, make_kernel([1, 3, 3, 1]).cuda(), pad=(1, 2))
        if not torch.isfinite(result).all():
            raise RuntimeError("The model's convolution returned non-finite values.")
        q = torch.randn(1, 8, 12, 128, device="cuda", dtype=torch.bfloat16)
        attention = F.scaled_dot_product_attention(q, q, q)
        if not torch.isfinite(attention).all():
            raise RuntimeError("BF16 attention returned non-finite values.")
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
        kept = nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
        if kept.tolist() != [0]:
            raise RuntimeError("torchvision CUDA NMS returned unexpected indices.")
        torch.cuda.synchronize()
    print("CUDA matrix multiplication, model convolution, BF16 attention and NMS: OK", flush=True)

    ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")
    subprocess.run([
        ffmpeg, "-v", "error", "-f", "lavfi", "-i",
        "color=c=black:s=32x32:r=25:d=0.08", "-c:v", "h264", "-f", "null", "-",
    ], check=True)
    print(f"FFmpeg H.264 encoding: OK ({ffmpeg})", flush=True)

    missing = [str(path.relative_to(root)) for path in (
        root / "pretrained_dir/flow_transformer.pth",
        root / "pretrained_dir/motion_autoencoder.pth",
        root / "pretrained_dir/wav2vec2-base-960h/config.json",
    ) if not path.is_file()]
    print("Environment checks passed.", flush=True)
    if missing:
        print("Checkpoints still needed for inference: " + ", ".join(missing))
    print("Full inference requires model checkpoints and is not part of this check.")


if __name__ == "__main__":
    main()
