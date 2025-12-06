# download_data.py
# author: Rebecca Rosette Nanfuka
# date: 2025-12-06

import click
import os
import zipfile
import requests
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.read_zip import read_zip


@click.command()
@click.option("--url", type=str, default="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", show_default=True, help="URL of dataset to be downloaded")
@click.option("--write_to", type=str, default="data/raw", show_default=True, help="Path to directory where raw data will be written to")
def main(url, write_to):
    """Downloads the Adult Income data file from the web to a local filepath."""
    try:
        read_zip(url, write_to)
    except FileNotFoundError:
        os.makedirs(write_to, exist_ok=True)
        read_zip(url, write_to)


if __name__ == "__main__":
    main()
