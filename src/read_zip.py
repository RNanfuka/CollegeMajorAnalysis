"""Utility helpers for downloading and extracting archives."""

from __future__ import annotations

from pathlib import Path
import zipfile

import requests


def read_zip(url: str, write_to: str) -> Path:
    """Download a file from ``url`` and extract it if it is a ZIP archive."""
    destination_dir = Path(write_to)
    destination_dir.mkdir(parents=True, exist_ok=True)

    target_name = Path(url).name or "dataset"
    download_path = destination_dir / target_name

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with download_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_obj.write(chunk)

    if zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path, "r") as archive:
            archive.extractall(destination_dir)
        download_path.unlink()

    return destination_dir
