from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def export_map_visuals(grid_npy: Path, output_png: Path) -> None:
    grid = np.load(grid_npy)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='gray')
    plt.title('SLAM Occupancy Grid (Phase 1)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
