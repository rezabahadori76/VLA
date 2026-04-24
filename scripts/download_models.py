from __future__ import annotations

import argparse
from pathlib import Path


def _download_hf_repo(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def _download_file(url: str, output_path: Path) -> None:
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[skip] {output_path} exists")
        return
    print(f"[download] {url} -> {output_path}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all model assets for Phase 1.")
    parser.add_argument("--root", type=Path, default=Path("models"), help="Output model root folder.")
    args = parser.parse_args()

    models_root = args.root
    models_root.mkdir(parents=True, exist_ok=True)

    # Grounding DINO
    _download_hf_repo("IDEA-Research/grounding-dino-tiny", models_root / "grounding_dino_tiny")

    # Qwen2-VL
    _download_hf_repo("Qwen/Qwen2-VL-2B-Instruct", models_root / "qwen2_vl_2b")

    # SAM checkpoint
    _download_file(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        models_root / "sam" / "sam_vit_h_4b8939.pth",
    )

    print("[done] model download complete")


if __name__ == "__main__":
    main()
