"""Download upstream inference checkpoints with retries and safe resuming."""

import fcntl
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import gdown
import requests
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars


ROOT = Path(__file__).resolve().parent / "pretrained_dir"
WAV2VEC_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
GOOGLE_FILES = (
    ("1a0ZqX9GMrkb3JFQgOBmUB0V-Bpwtv4Hp", "motion_autoencoder.pth"),
    ("1lchvHiDka0bBQ29bdY6lHxCyNGbGzvS_", "flow_transformer.pth"),
)


class ResumeRejected(RuntimeError):
    pass


def retry(operation):
    for attempt in range(1, 6):
        try:
            return operation()
        except Exception as error:
            print(f"Attempt {attempt}/5 failed: {type(error).__name__}", flush=True)
            if attempt == 5:
                raise
            time.sleep(5)


def download_google(identifier, filename):
    output = ROOT / filename
    if output.is_file():
        print(f"Already downloaded: {filename} ({output.stat().st_size:,} bytes)", flush=True)
        return
    original_get = requests.Session.get

    def checked_get(session, url, **kwargs):
        kwargs.setdefault("timeout", (20, 120))
        response = original_get(session, url, **kwargs)
        requested_range = (kwargs.get("headers") or {}).get("Range")
        if requested_range:
            offset = requested_range.removeprefix("bytes=").split("-")[0]
            content_range = response.headers.get("Content-Range", "")
            if response.status_code == 200:
                response.close()
                raise ResumeRejected("Server ignored the download offset.")
            response.raise_for_status()
            if response.status_code != 206 or not content_range.startswith(f"bytes {offset}-"):
                response.close()
                raise ResumeRejected("Unexpected response to a resumed download.")
            print(f"Resuming {filename}: {content_range}", flush=True)
        return response

    def download():
        print(f"Downloading: {filename}", flush=True)
        requests.Session.get = checked_get
        try:
            result = gdown.download(
                id=identifier, output=str(output), quiet=True,
                resume=True, use_cookies=False,
            )
        except ResumeRejected:
            for partial in ROOT.glob(filename + "*.part"):
                partial.rename(partial.with_name(partial.name + f".{time.time_ns()}.backup"))
            raise
        finally:
            requests.Session.get = original_get
        if result is None:
            raise RuntimeError(f"Google Drive download failed: {filename}")
        print(f"Downloaded: {filename} ({output.stat().st_size:,} bytes)", flush=True)

    retry(download)


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for identifier, filename in GOOGLE_FILES:
            download_google(identifier, filename)
        print(f"Downloading Wav2Vec2 at revision {WAV2VEC_REVISION}", flush=True)
        disable_progress_bars()
        retry(lambda: snapshot_download(
            repo_id="facebook/wav2vec2-base-960h", revision=WAV2VEC_REVISION,
            local_dir=str(ROOT / "wav2vec2-base-960h"), max_workers=4,
            allow_patterns=["*.json", "model.safetensors", "README.md", ".gitattributes"],
        ))
        print("All inference checkpoint downloads finished.", flush=True)


if __name__ == "__main__":
    main()
