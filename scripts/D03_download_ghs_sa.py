#!/usr/bin/env python3
import os, zipfile, requests
from pathlib import Path

BASE_DIR = Path('/home/jupyter-daniela/CAF/data/GHS')
JRC_BASE = 'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL'

LAYERS = {
    'BUILT_S': {
        'years':   [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030],
        'dataset': 'GHS_BUILT_S_GLOBE_R2023A',
        'prefix':  'GHS_BUILT_S',
    },
    'BUILT_V': {
        'years':   [1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030],
        'dataset': 'GHS_BUILT_V_GLOBE_R2023A',
        'prefix':  'GHS_BUILT_V',
    },
    'POP': {
        'years':   [1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030],
        'dataset': 'GHS_POP_GLOBE_R2023A',
        'prefix':  'GHS_POP',
    },
}

TILES = [
    ('R9',  'C11'), ('R9',  'C12'), ('R9',  'C13'),
    ('R10', 'C11'), ('R10', 'C12'),
    ('R11', 'C11'), ('R11', 'C12'),
    ('R12', 'C11'),
]

def download_tile(layer, year, row, col, out_dir):
    cfg      = LAYERS[layer]
    name     = f"{cfg['prefix']}_E{year}_GLOBE_R2023A_54009_100_V1_0_{row}_{col}"
    tif_path = out_dir / f"{name}.tif"
    zip_path = out_dir / f"{name}.zip"

    if tif_path.exists():
        print(f"  [skip] {name}.tif")
        return True

    url = (f"{JRC_BASE}/{cfg['dataset']}/{cfg['prefix']}_E{year}_GLOBE_R2023A_54009_100"
           f"/V1-0/tiles/{name}.zip")
    print(f"  [dl] {name} …", end=' ', flush=True)
    try:
        r = requests.get(url, timeout=120, stream=True)
        if r.status_code == 404:
            print("not found (tile may not exist)")
            return False
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out_dir)
        zip_path.unlink(missing_ok=True)
        size = tif_path.stat().st_size / 1e6 if tif_path.exists() else 0
        print(f"{size:.1f} MB")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        zip_path.unlink(missing_ok=True)
        return False

total, skipped, failed = 0, 0, 0
for layer, cfg in LAYERS.items():
    for year in cfg['years']:
        out_dir = BASE_DIR / layer / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        for row, col in TILES:
            total += 1
            ok = download_tile(layer, year, row, col, out_dir)
            if not ok:
                failed += 1

print(f"\nDone: {total - failed} OK, {failed} failed/missing")
