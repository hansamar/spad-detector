import numpy as np


def background_base_series(t, scene_stray_rate, drift_depth=0.15, drift_hz=3.0):
    """Scene ambient background time series in cps/pixel."""
    t = np.asarray(t, dtype=np.float64)
    drift = 1.0 + drift_depth * np.cos(2.0 * np.pi * drift_hz * t + 0.5)
    return (scene_stray_rate * np.maximum(drift, 0.0)).astype(np.float64)


def background_spatial_map(roi_h, roi_w, sigma=0.0, gradient_x=0.0, gradient_y=0.0, rng=None):
    """Scene ambient spatial factor with unit mean."""
    if sigma > 0 and rng is not None:
        bg_spatial = 1.0 + rng.normal(0, sigma, size=(roi_h, roi_w))
    else:
        bg_spatial = np.ones((roi_h, roi_w), dtype=np.float64)

    col_coords = np.arange(roi_w, dtype=np.float64) - (roi_w - 1) / 2.0
    row_coords = np.arange(roi_h, dtype=np.float64) - (roi_h - 1) / 2.0
    gradient = gradient_x * col_coords[np.newaxis, :] + gradient_y * row_coords[:, np.newaxis]
    bg_spatial = bg_spatial + gradient

    mean_val = np.mean(bg_spatial)
    if mean_val > 0:
        bg_spatial = bg_spatial / mean_val

    return bg_spatial.astype(np.float32)


def background_cube_series(bg_base_t, bg_spatial):
    """Build the scene ambient background cube in cps."""
    bg_base_t = np.asarray(bg_base_t, dtype=np.float64)
    bg_spatial = np.asarray(bg_spatial, dtype=np.float64)
    return bg_base_t[:, None, None] * bg_spatial[None, :, :]
