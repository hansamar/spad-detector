"""Geometry helpers for near-range SPAD active imaging scenes."""

from __future__ import annotations

import numpy as np


def make_time_axis(observation_time_s, sample_rate_hz):
    n_frames = int(round(observation_time_s * sample_rate_hz))
    dt = 1.0 / sample_rate_hz
    t = np.arange(n_frames) * dt
    return t, dt, n_frames


def range_series(t, R0, v=0.0, a=0.0):
    return R0 + v * t + 0.5 * a * t**2


def sun_direction_series(t, elevation_deg, azimuth_deg, rate_deg_per_s=0.0):
    az_rad = np.deg2rad(azimuth_deg + rate_deg_per_s * t)
    el_rad = np.deg2rad(elevation_deg)

    cos_el = np.cos(el_rad)
    sin_el = np.sin(el_rad)
    x = cos_el * np.cos(az_rad)
    y = cos_el * np.sin(az_rad)
    z = np.full_like(t, sin_el)
    return np.stack([x, y, z], axis=-1)


def line_of_sight_series(t, elevation_deg=0.0, azimuth_deg=0.0, rate_deg_per_s=0.0):
    az_rad = np.deg2rad(azimuth_deg + rate_deg_per_s * t)
    el_rad = np.deg2rad(elevation_deg)

    cos_el = np.cos(el_rad)
    sin_el = np.sin(el_rad)
    x = cos_el * np.cos(az_rad)
    y = cos_el * np.sin(az_rad)
    z = np.full_like(t, sin_el)
    return np.stack([x, y, z], axis=-1)


def phase_angle_series(sun_unit, los_unit):
    cos_phase = np.sum(sun_unit * los_unit, axis=-1)
    return np.arccos(np.clip(cos_phase, -1.0, 1.0))


def boresight_offset_series(
    t,
    mode,
    initial_x_urad=0.0,
    initial_y_urad=0.0,
    angular_rate_urad_s_x=0.0,
    angular_rate_urad_s_y=0.0,
    tracking_residual_sigma_urad=0.0,
    reacquire_enabled=False,
    reacquire_start_fraction=0.35,
    reacquire_end_fraction=0.55,
    fov_urad=50.0,
    edge_entry_mode="manual",
    rng=None,
    trajectory_name="",
):
    """Generate focal-plane boresight offsets for current near-range scenes."""
    t = np.asarray(t, dtype=np.float64)
    n_frames = t.size
    if n_frames == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    mode = (mode or "centered_roi").lower()
    half_fov = max(float(fov_urad) / 2.0, 1e-6)
    if rng is None:
        rng = np.random.default_rng(1234)

    if mode == "centered_roi":
        off_x = np.full(n_frames, initial_x_urad, dtype=np.float64)
        off_y = np.full(n_frames, initial_y_urad, dtype=np.float64)
    elif trajectory_name:
        off_x, off_y = _named_boresight_trajectory(
            t,
            trajectory_name=str(trajectory_name),
            half_fov=half_fov,
        )
    else:
        if mode in ("moving_target_in_fov", "search_and_reacquire") and edge_entry_mode == "random_edge":
            side = int(rng.integers(0, 4))
            margin = 0.82 * half_fov
            if side == 0:
                initial_x_urad, initial_y_urad = (-margin, rng.uniform(-0.3, 0.3) * half_fov)
            elif side == 1:
                initial_x_urad, initial_y_urad = (margin, rng.uniform(-0.3, 0.3) * half_fov)
            elif side == 2:
                initial_x_urad, initial_y_urad = (rng.uniform(-0.3, 0.3) * half_fov, -margin)
            else:
                initial_x_urad, initial_y_urad = (rng.uniform(-0.3, 0.3) * half_fov, margin)

        off_x = initial_x_urad + angular_rate_urad_s_x * t
        off_y = initial_y_urad + angular_rate_urad_s_y * t

    if tracking_residual_sigma_urad > 0:
        off_x = off_x + rng.normal(0.0, tracking_residual_sigma_urad, size=n_frames)
        off_y = off_y + rng.normal(0.0, tracking_residual_sigma_urad, size=n_frames)

    reacquire_flag = np.zeros(n_frames, dtype=np.float32)
    if mode == "search_and_reacquire" and reacquire_enabled:
        start_idx = int(np.clip(round(reacquire_start_fraction * n_frames), 0, n_frames - 1))
        end_idx = int(np.clip(round(reacquire_end_fraction * n_frames), start_idx + 1, n_frames))
        dominant_axis_x = abs(angular_rate_urad_s_x) >= abs(angular_rate_urad_s_y)
        offset_push = 1.2 * half_fov
        if dominant_axis_x:
            sign = 1.0 if (angular_rate_urad_s_x or initial_x_urad or 1.0) >= 0 else -1.0
            off_x[start_idx:end_idx] = sign * offset_push
        else:
            sign = 1.0 if (angular_rate_urad_s_y or initial_y_urad or 1.0) >= 0 else -1.0
            off_y[start_idx:end_idx] = sign * offset_push
        reacquire_flag[end_idx:] = 1.0

    radial = np.sqrt(off_x**2 + off_y**2)
    in_fov = (radial <= half_fov).astype(np.float32)
    return off_x.astype(np.float64), off_y.astype(np.float64), in_fov, reacquire_flag


def _interp_polyline(s, points):
    points = np.asarray(points, dtype=np.float64)
    xp = np.linspace(0.0, 1.0, points.shape[0])
    x = np.interp(s, xp, points[:, 0])
    y = np.interp(s, xp, points[:, 1])
    return x, y


def _named_boresight_trajectory(t, trajectory_name, half_fov):
    """Generate deterministic near-range target trajectories."""
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    s = t / max(float(t[-1]), 1e-9)
    local_margin = 0.76 * float(half_fov)

    if trajectory_name == "linear_center_sweep":
        x = -0.48 * half_fov + 0.30 * half_fov * s
        y = -0.05 * half_fov * np.ones_like(s)
    elif trajectory_name == "linear_diagonal_rise":
        x = -0.46 * half_fov + 0.28 * half_fov * s
        y = 0.18 * half_fov - 0.22 * half_fov * s
    elif trajectory_name == "linear_diagonal_fall":
        x = -0.46 * half_fov + 0.28 * half_fov * s
        y = -0.18 * half_fov + 0.22 * half_fov * s
    elif trajectory_name == "sine_soft":
        x = -0.44 * half_fov + 0.30 * half_fov * s
        y = 0.16 * half_fov * np.sin(2.0 * np.pi * s)
    elif trajectory_name == "sine_wide":
        x = -0.42 * half_fov + 0.24 * half_fov * s
        y = 0.22 * half_fov * np.sin(2.0 * np.pi * s + np.pi / 6.0)
    elif trajectory_name == "arc_upper":
        x = -0.42 * half_fov + 0.26 * half_fov * s
        y = 0.18 * half_fov * np.sin(np.pi * s)
    elif trajectory_name == "arc_lower":
        x = -0.42 * half_fov + 0.26 * half_fov * s
        y = -0.18 * half_fov * np.sin(np.pi * s)
    elif trajectory_name == "arc_blue_demo":
        local_margin = 0.95 * float(half_fov)
        x = -0.90 * half_fov + 1.80 * half_fov * s
        y = -0.14 * half_fov + 0.14 * half_fov * np.sin(np.pi * s)
    elif trajectory_name == "circle_small":
        x = -0.06 * half_fov + 0.18 * half_fov * np.cos(2.0 * np.pi * s)
        y = 0.18 * half_fov * np.sin(2.0 * np.pi * s)
    elif trajectory_name == "ellipse_tilted":
        x = -0.10 * half_fov + 0.24 * half_fov * np.cos(2.0 * np.pi * s + 0.3)
        y = 0.12 * half_fov * np.sin(2.0 * np.pi * s)
    elif trajectory_name == "polyline_s_bend":
        x, y = _interp_polyline(
            s,
            [
                (-0.46 * half_fov, 0.12 * half_fov),
                (-0.30 * half_fov, -0.16 * half_fov),
                (-0.12 * half_fov, 0.14 * half_fov),
                (0.04 * half_fov, -0.10 * half_fov),
            ],
        )
    else:
        raise ValueError(f"Unsupported near-range trajectory_name: {trajectory_name}")

    x = np.clip(x, -local_margin, local_margin)
    y = np.clip(y, -local_margin, local_margin)
    return x.astype(np.float64), y.astype(np.float64)
