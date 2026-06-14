"""Download MapBiomas Brazil collection tifs (coverage_lclu and
agriculture_agricultural_use_second_crop) for 2020-2024 into ./mapbiomas/.

The download links follow the public GCS pattern returned by the
MapBiomas export API (POST https://prd.plataforma.mapbiomas.org/api/v1/brazil/maps/export):

    https://storage.googleapis.com/mapbiomas-downloads/public/brazil/maps/{uuid}/{year}_{type}_1-1-1_{uuid}.tif

The UUIDs below were obtained from that API response (agriculture_agricultural_use_second_crop)
and from the filenames of files already downloaded via the platform (coverage_lclu).
"""

import os
import requests
from tqdm import tqdm

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapbiomas")

BASE_URL = "https://storage.googleapis.com/mapbiomas-downloads/public/brazil/maps"

DATASETS = {
    "coverage_lclu": {
        2020: "ac1d9d62-6d91-4564-bc89-452c369af273",
        2021: "a4abb8ef-433d-45c0-9e1b-56141b9a83b7",
        2022: "2361a4a7-8905-4672-b3b9-498a52407de7",
        2023: "2242342e-4c61-44e4-83ac-2a2e7e23b148",
        2024: "e04830ad-d1b1-4739-b27b-ffea882f0d77",
    },
    "agriculture_agricultural_use_second_crop": {
        2020: "6b693804-82ed-4e1e-98ce-472b41a8d0b4",
        2021: "1a0f0412-2bb2-4a8f-b87a-8333e58ff076",
        2022: "138bb332-92ab-4e72-9ea0-1b915472fe5e",
        2023: "ce1ce9ae-9432-454a-ae29-6bf44860176c",
        2024: "f968dc13-8eb5-4c84-a28f-d603591dd6b6",
    },
}


def build_links():
    links = []
    for dataset_type, years in DATASETS.items():
        for year, uuid in years.items():
            filename = f"{year}_{dataset_type}_1-1-1_{uuid}.tif"
            url = f"{BASE_URL}/{uuid}/{filename}"
            links.append((filename, url))
    return links


def download(filename, url, dest_dir):
    dest_path = os.path.join(dest_dir, filename)

    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    remote_size = int(head.headers.get("content-length", 0))

    resume_from = 0
    mode = "wb"
    if os.path.exists(dest_path):
        local_size = os.path.getsize(dest_path)
        if local_size == remote_size:
            print(f"Skipping {filename} (already downloaded, {local_size} bytes)")
            return
        if local_size < remote_size:
            resume_from = local_size
            mode = "ab"

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, mode) as f, tqdm(
            total=remote_size,
            initial=resume_from,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename, url in build_links():
        download(filename, url, OUTPUT_DIR)


if __name__ == "__main__":
    main()
