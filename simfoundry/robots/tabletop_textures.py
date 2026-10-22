# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Procedural tabletop textures for domain randomization (numpy + PIL only, no simulator).

Wood: anisotropic periodic noise (long streaks along x) mapped through a light-to-dark wood
gradient, plus fine fibres. The palette is centred on a light oak / beech office table
(RGB ~ 215, 180, 130) with enough spread to cover darker and redder woods. Textures tile, so the
simulator can repeat them over a large plane at any scale.
"""

import os

import numpy as np
from PIL import Image

# (light, dark) endpoints of the grain gradient, 0-255 RGB.
WOOD_PALETTES = {
    "light_oak": ((226, 194, 148), (190, 150, 100)),
    "beech": ((232, 200, 158), (200, 160, 112)),
    "honey_pine": ((222, 182, 120), (176, 128, 70)),
    "ash": ((218, 196, 164), (184, 158, 120)),
    "walnut": ((150, 108, 72), (96, 62, 38)),
    "cherry": ((176, 112, 76), (128, 72, 44)),
    "pale_birch": ((238, 214, 178), (214, 184, 142)),
}


def _periodic_noise(rng, size, stretch):
    """Tileable noise, correlated @stretch times longer along x than y."""
    fx = np.fft.fftfreq(size)[None, :]
    fy = np.fft.fftfreq(size)[:, None]
    spectrum = 1.0 / (1e-3 + (fx * stretch) ** 2 * 0 + np.sqrt((fx * stretch) ** 2 + fy**2)) ** 1.6
    spectrum[0, 0] = 0.0
    phase = np.exp(2j * np.pi * rng.random((size, size)))
    noise = np.real(np.fft.ifft2(spectrum * phase))
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)


def make_wood_texture(palette, seed, size=1024):
    """(size, size, 3) uint8 wood texture, deterministic in @seed."""
    rng = np.random.default_rng(seed)
    light, dark = (np.asarray(c, float) for c in WOOD_PALETTES[palette])
    stretch = rng.uniform(8.0, 20.0)
    grain = _periodic_noise(rng, size, stretch)             # long streaks
    fibre = _periodic_noise(rng, size, stretch * 3.0)        # fine fibres
    swirl = _periodic_noise(rng, size, 2.0)                  # slow large-scale tone drift
    t = np.clip(0.55 * grain + 0.25 * fibre + 0.20 * swirl, 0, 1)
    t = (t - t.mean()) * rng.uniform(1.4, 2.2) + 0.5          # contrast of the grain
    t = np.clip(t, 0, 1)[..., None]
    rgb = light * (1 - t) + dark * t
    return np.clip(rgb, 0, 255).astype(np.uint8)


def generate_wood_textures(out_dir, per_palette=2, size=1024):
    """Write wood_<palette>_<i>.png files (skipping existing ones); return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for p_idx, palette in enumerate(WOOD_PALETTES):
        for i in range(per_palette):
            path = os.path.join(out_dir, f"wood_{palette}_{i}.png")
            if not os.path.exists(path):
                Image.fromarray(make_wood_texture(palette, seed=1000 * p_idx + i, size=size)).save(path)
            paths.append(path)
    return paths


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "assets/textures/tabletop"
    for p in generate_wood_textures(out):
        print(p)
