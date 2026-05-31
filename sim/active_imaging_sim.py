from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sim.attitude import (
    face_normals_in_inertial,
    precession_axis_series,
    simple_body_model,
    simple_body_vertices,
    spin_rotation_matrix_series,
)
from sim.background import background_spatial_map
from sim.config import SimParams
from sim.detector_presets import preset_summary
from sim.detector import apply_dead_time_rate, make_dark_map, make_pde_map, sample_poisson_counts_accelerated
from sim.geometry import (
    boresight_offset_series,
    line_of_sight_series,
    make_time_axis,
    phase_angle_series,
    range_series,
    sun_direction_series,
)
from sim.imaging import (
    _apply_separable_blur,
    centroid_from_off_axis_series,
    jitter_series_gauss_markov,
    jitter_series_sinusoidal,
    jitter_series_white,
    projected_extent_series_from_vertices,
    signal_distribution_cube,
    signal_distribution_cube_projected,
    target_centroid_series,
)
from sim.physics import (
    aperture_area,
    atmospheric_transmittance,
    laser_target_detected_rate_cps,
    make_visibility_mask,
    modulation_series,
    target_detected_rate_cps,
)
from sim.reflectance import target_lightcurve_attitude_driven


def _normalized(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


def _make_sun_from_phase(los_unit_t: np.ndarray, phase_deg: float, phase_rate_deg_per_s: float, t: np.ndarray) -> np.ndarray:
    phase_rad_t = np.deg2rad(phase_deg + phase_rate_deg_per_s * t)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    alt_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    basis_1 = np.cross(los_unit_t, ref[None, :])
    degenerate = np.linalg.norm(basis_1, axis=-1) < 1e-8
    if np.any(degenerate):
        basis_1[degenerate] = np.cross(los_unit_t[degenerate], alt_ref[None, :])
    basis_1 = _normalized(basis_1)
    basis_2 = _normalized(np.cross(basis_1, los_unit_t))
    sun_unit_t = (
        np.cos(phase_rad_t)[:, None] * los_unit_t
        + np.sin(phase_rad_t)[:, None] * basis_1
        + 0.05 * np.sin(0.13 * phase_rad_t)[:, None] * basis_2
    )
    return _normalized(sun_unit_t)


def _external_series(value, n_frames: int, name: str, *, ndim: int = 1) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing recorded-flight geometry field: {name}")
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape[0] != n_frames:
        raise ValueError(f"{name} length {arr.shape[0]} does not match simulation frame count {n_frames}.")
    if ndim == 1 and arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional series.")
    if ndim == 2 and (arr.ndim != 2 or arr.shape[1] != 3):
        raise ValueError(f"{name} must have shape [n_frames, 3].")
    return arr


def _build_external_geometry(t: np.ndarray, params: SimParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    n_frames = int(t.size)
    range_m = _external_series(params.geometry.external_range_m, n_frames, "external_range_m")
    sun_unit = _normalized(_external_series(params.geometry.external_sun_unit, n_frames, "external_sun_unit", ndim=2))
    los_unit = _normalized(_external_series(params.geometry.external_los_unit, n_frames, "external_los_unit", ndim=2))
    if params.geometry.external_phase_angle_rad is None:
        phase_angle = phase_angle_series(sun_unit, los_unit)
    else:
        phase_angle = _external_series(params.geometry.external_phase_angle_rad, n_frames, "external_phase_angle_rad")
    assumptions = [
        "Manual recorded flight uses frontend-recorded drone trajectory samples.",
        "Recorded target position drives range, line-of-sight, and detector FOV screening.",
        "Optical properties remain assumed target parameters rather than measured photometry.",
    ]
    return range_m, sun_unit, los_unit, phase_angle, assumptions


def _build_geometry(t: np.ndarray, params: SimParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    trajectory_name = getattr(params.geometry, "trajectory_name", "")
    if trajectory_name == "manual_recorded_flight":
        return _build_external_geometry(t, params)

    R_t = range_series(
        t,
        R0=params.target.target_range_m,
        v=params.target.target_radial_velocity_mps,
        a=params.target.target_radial_accel_mps2,
    )

    if params.simulation_tier == "baseline_empirical":
        sun_unit_t = sun_direction_series(
            t,
            elevation_deg=params.geometry.sun_elevation_deg,
            azimuth_deg=params.geometry.sun_azimuth_deg,
            rate_deg_per_s=params.geometry.sun_rate_deg_per_s,
        )
        los_unit_t = line_of_sight_series(
            t,
            elevation_deg=params.geometry.los_elevation_deg,
            azimuth_deg=params.geometry.los_azimuth_deg,
            rate_deg_per_s=params.geometry.los_rate_deg_per_s,
        )
        assumptions = [
            "Baseline empirical tier uses decoupled range, line-of-sight, and sun-direction series.",
            "Background terms are specified as empirical rates per pixel.",
        ]
        return R_t, sun_unit_t, los_unit_t, phase_angle_series(sun_unit_t, los_unit_t), assumptions

    view_rate = np.deg2rad(max(0.02, abs(params.geometry.los_rate_deg_per_s) or 0.08))
    view_phase = view_rate * t
    target_vec = np.stack(
        [
            np.cos(view_phase),
            np.sin(view_phase),
            0.18 * np.sin(0.37 * view_phase + 0.4),
        ],
        axis=-1,
    )
    los_unit_t = _normalized(-target_vec)

    if params.geometry.phase_angle_override_deg >= 0:
        sun_unit_t = _make_sun_from_phase(
            los_unit_t,
            params.geometry.phase_angle_override_deg,
            params.geometry.phase_angle_rate_deg_per_s,
            t,
        )
        assumptions = [
            "Physics-informed tier uses synthetic relative orbit geometry with user-controlled phase angle.",
            "Phase-angle override is intended for feasibility studies and detector-boundary scans.",
            "Background components respond to phase angle, boresight, and stray-light coupling.",
        ]
    else:
        sun_phase = view_phase * 0.71 + np.deg2rad(params.geometry.sun_azimuth_deg)
        sun_unit_t = _normalized(
            np.stack(
                [
                    np.cos(sun_phase),
                    0.6 * np.sin(sun_phase),
                    np.sin(np.deg2rad(params.geometry.sun_elevation_deg)) + 0.15 * np.cos(0.25 * view_phase),
                ],
                axis=-1,
            )
        )
        assumptions = [
            "Physics-informed tier couples target-sensor-sun geometry through one synthetic relative orbit track.",
            "This is geometry-informed simulation rather than full flight-dynamics propagation.",
            "Background components respond to phase angle, boresight, and stray-light coupling.",
        ]
    if np.any(R_t <= 0):
        warnings.append("Range trajectory crossed non-physical values and was clipped.")
        R_t = np.maximum(R_t, 1.0)
    return R_t, sun_unit_t, los_unit_t, phase_angle_series(sun_unit_t, los_unit_t), assumptions + warnings


def _detector_geometry(params: SimParams) -> tuple[float, float, float]:
    total_fov_urad = max(float(params.optical.detector_fov_urad), 1e-6)
    pixel_axis = float(max(params.image.roi_w, params.image.roi_h))
    pixel_ifov_urad = total_fov_urad / max(pixel_axis, 1.0)
    off_axis_urad = float(
        np.hypot(
            getattr(params.optical, "detector_off_axis_urad_x", 0.0),
            getattr(params.optical, "detector_off_axis_urad_y", 0.0),
        )
    )
    return total_fov_urad, pixel_ifov_urad, off_axis_urad


def _target_signal(
    t: np.ndarray,
    R_t: np.ndarray,
    sun_unit_t: np.ndarray,
    los_unit_t: np.ndarray,
    phase_angle_t: np.ndarray,
    params: SimParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray]:
    aperture = aperture_area(params.optical.aperture_diameter_m)
    illumination_mode = str(params.illumination.mode or "laser_plus_solar").lower()
    solar_enabled = illumination_mode in {"solar", "laser_plus_solar"}
    laser_enabled = illumination_mode in {"laser", "laser_plus_solar"}
    solar_rate_cps_t = np.zeros_like(t, dtype=np.float64)
    laser_rate_cps_t = np.zeros_like(t, dtype=np.float64)

    active_shape = str(params.target.body_shape or "") in {"sphere", "blade_strip", "drone_quad"}
    use_attitude = params.lightcurve_mode == "attitude_driven" or (
        params.simulation_tier == "physics_informed" and params.target.spin_hz > 0
    )
    use_attitude = use_attitude and not active_shape
    if use_attitude:
        spin_hz = params.target.spin_hz if params.target.spin_hz > 0 else max(params.target.tumbling_hz / 500.0, 0.5)
        spin_axis_init = np.array(
            [
                np.sin(np.deg2rad(params.target.spin_axis_elevation_deg))
                * np.cos(np.deg2rad(params.target.spin_axis_azimuth_deg)),
                np.sin(np.deg2rad(params.target.spin_axis_elevation_deg))
                * np.sin(np.deg2rad(params.target.spin_axis_azimuth_deg)),
                np.cos(np.deg2rad(params.target.spin_axis_elevation_deg)),
            ]
        )
        axis_t = (
            precession_axis_series(t, spin_axis_init, params.target.precession_hz, cone_angle_deg=12.0)
            if params.target.precession_hz > 0
            else np.broadcast_to(_normalized(spin_axis_init[None, :]), (t.size, 3)).copy()
        )

        faces = simple_body_model(shape=params.target.body_shape)
        total_area = sum(face["area"] for face in faces)
        if total_area > 0:
            scale = params.target.target_area_m2 / total_area
            for face in faces:
                face["area"] *= scale
                face["rho"] = params.target.target_reflectivity
                face["specular"] = max(face["specular"], params.target.specular_fraction)

        R_bi_all = spin_rotation_matrix_series(t, spin_hz, axis_t, phase0=params.target.phase1)
        normals_t = face_normals_in_inertial(R_bi_all, faces)
        attitude_scale_t, attitude_solar_rate_cps_t = target_lightcurve_attitude_driven(
            faces=faces,
            normals_t=normals_t,
            sun_unit_t=sun_unit_t,
            los_unit_t=los_unit_t,
            range_t=R_t,
            solar_irradiance=params.target.solar_irradiance_w_m2_nm,
            aperture_area_m2=aperture,
            filter_bw_nm=params.optical.filter_bandwidth_nm,
            wavelength_nm=params.optical.wavelength_nm,
            receiver_eff=params.optical.receiver_efficiency,
            quantum_eff=params.optical.quantum_efficiency,
            phase_function_scale=params.target.phase_function_scale,
            specular_width_deg=params.target.specular_width_deg,
        )
        if solar_enabled:
            solar_rate_cps_t = attitude_solar_rate_cps_t
        if laser_enabled:
            laser_attitude_scale_t, _ = target_lightcurve_attitude_driven(
                faces=faces,
                normals_t=normals_t,
                sun_unit_t=los_unit_t,
                los_unit_t=los_unit_t,
                range_t=R_t,
                solar_irradiance=1.0,
                aperture_area_m2=aperture,
                filter_bw_nm=params.optical.filter_bandwidth_nm,
                wavelength_nm=params.optical.wavelength_nm,
                receiver_eff=params.optical.receiver_efficiency,
                quantum_eff=params.optical.quantum_efficiency,
                phase_function_scale=params.target.phase_function_scale,
                specular_width_deg=params.target.specular_width_deg,
            )
            laser_rate_cps_t = laser_target_detected_rate_cps(
                laser_mode=params.illumination.laser_mode,
                laser_average_power_w=params.illumination.laser_average_power_w,
                laser_pulse_energy_j=params.illumination.laser_pulse_energy_j,
                laser_repetition_frequency_hz=params.illumination.laser_repetition_frequency_hz,
                transmitter_divergence_mrad=params.illumination.transmitter_divergence_mrad,
                target_area_m2=params.target.target_area_m2,
                target_reflectivity=params.target.target_reflectivity,
                phase_function_scale=params.target.phase_function_scale,
                range_m=R_t,
                aperture_diameter_m=params.optical.aperture_diameter_m,
                wavelength_nm=params.optical.wavelength_nm,
                receiver_efficiency=params.optical.receiver_efficiency,
                quantum_efficiency=params.optical.quantum_efficiency,
            ) * np.maximum(laser_attitude_scale_t, 0.0)
        target_rate_cps_t = solar_rate_cps_t + laser_rate_cps_t
        peak_rate = float(np.max(target_rate_cps_t))
        signal_scale_t = target_rate_cps_t / peak_rate if peak_rate > 0 else np.zeros_like(target_rate_cps_t)
        truth_freq_hz = spin_hz
        truth_precession_hz = params.target.precession_hz
        projection_rotation_t = R_bi_all
    else:
        base_mod = modulation_series(
            t=t,
            tumbling_hz=params.target.tumbling_hz,
            m1=params.target.modulation_depth,
            m2=params.target.harmonic2_depth,
            m3=params.target.harmonic3_depth,
            p1=params.target.phase1,
            p2=params.target.phase2,
            p3=params.target.phase3,
            slow_depth=params.target.slow_envelope_depth,
            slow_hz=params.target.slow_envelope_hz,
        )
        if params.simulation_tier == "physics_informed":
            phase_gain = 0.45 + 0.55 * np.clip(np.cos(phase_angle_t / 2.0), 0.0, 1.0)
            solar_mod = base_mod * phase_gain
        else:
            solar_mod = base_mod
        if solar_enabled:
            base_solar_rate_cps = target_detected_rate_cps(
                irradiance_w_m2_nm=params.target.solar_irradiance_w_m2_nm,
                target_area_m2=params.target.target_area_m2,
                target_reflectivity=params.target.target_reflectivity,
                phase_function_scale=params.target.phase_function_scale,
                range_m=params.target.reference_range_m,
                aperture_diameter_m=params.optical.aperture_diameter_m,
                filter_bandwidth_nm=params.optical.filter_bandwidth_nm,
                wavelength_nm=params.optical.wavelength_nm,
                receiver_efficiency=params.optical.receiver_efficiency,
                quantum_efficiency=params.optical.quantum_efficiency,
            )
            range_factor = (params.target.reference_range_m / np.maximum(R_t, 1.0)) ** 2
            solar_rate_cps_t = base_solar_rate_cps * solar_mod * range_factor
        if laser_enabled:
            laser_rate_cps_t = laser_target_detected_rate_cps(
                laser_mode=params.illumination.laser_mode,
                laser_average_power_w=params.illumination.laser_average_power_w,
                laser_pulse_energy_j=params.illumination.laser_pulse_energy_j,
                laser_repetition_frequency_hz=params.illumination.laser_repetition_frequency_hz,
                transmitter_divergence_mrad=params.illumination.transmitter_divergence_mrad,
                target_area_m2=params.target.target_area_m2,
                target_reflectivity=params.target.target_reflectivity,
                phase_function_scale=params.target.phase_function_scale,
                range_m=R_t,
                aperture_diameter_m=params.optical.aperture_diameter_m,
                wavelength_nm=params.optical.wavelength_nm,
                receiver_efficiency=params.optical.receiver_efficiency,
                quantum_efficiency=params.optical.quantum_efficiency,
            ) * base_mod
        target_rate_cps_t = solar_rate_cps_t + laser_rate_cps_t
        peak_rate = float(np.max(target_rate_cps_t))
        signal_scale_t = target_rate_cps_t / peak_rate if peak_rate > 0 else np.zeros_like(target_rate_cps_t)
        truth_freq_hz = params.target.tumbling_hz
        truth_precession_hz = params.target.slow_envelope_hz
        projection_rotation_t = np.broadcast_to(np.eye(3, dtype=np.float64), (t.size, 3, 3)).copy()

    harmonic_truth_strength = (
        params.target.modulation_depth
        + 0.6 * params.target.harmonic2_depth
        + 0.35 * params.target.harmonic3_depth
        + 0.2 * max(params.target.specular_fraction, params.target.glint_probability)
    )
    return (
        signal_scale_t,
        target_rate_cps_t,
        solar_rate_cps_t,
        laser_rate_cps_t,
        truth_freq_hz,
        truth_precession_hz,
        harmonic_truth_strength,
        projection_rotation_t,
    )


def _background_series(
    t: np.ndarray,
    phase_angle_t: np.ndarray,
    los_unit_t: np.ndarray,
    params: SimParams,
) -> tuple[np.ndarray, np.ndarray]:
    drift = 1.0 + params.background.temporal_drift_depth * np.cos(
        2.0 * np.pi * params.background.temporal_drift_hz * t + 0.5
    )
    drift = np.maximum(drift, 0.1)
    scene_rate = np.full_like(t, params.background.scene_stray_rate_cps_per_pixel, dtype=np.float64)
    solar_elongation = np.clip(np.cos(phase_angle_t), -1.0, 1.0)
    geometry = 0.35 + 0.65 * np.clip(0.5 + 0.5 * solar_elongation, 0.0, 1.0)
    scene_stray = scene_rate * geometry * drift
    return scene_stray, scene_stray


def _background_scaling(params: SimParams, pixel_ifov_urad: float) -> float:
    _ = pixel_ifov_urad
    # Background rates are already detector-side cps/pixel. Scaling them again
    # by aperture, IFOV, bandwidth, and QE double-counts the optical chain.
    return 1.0


def _shot_noise_snr_db(total_signal: float, total_noise: float) -> float:
    """Aggregate photon-counting SNR for independent Poisson signal and noise."""
    denominator = np.sqrt(max(total_signal + total_noise, 1e-24))
    return float(20.0 * np.log10(max(total_signal, 1e-12) / denominator))


def _event_stream_from_counts(
    counts_cube: np.ndarray,
    dt: float,
    timing_jitter_ns: float,
    tdc_bin_width_ns: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames, roi_h, roi_w = counts_cube.shape
    event_times: list[np.ndarray] = []
    event_pixels: list[np.ndarray] = []

    for frame_idx in range(n_frames):
        frame_counts = counts_cube[frame_idx]
        active_pixels = np.argwhere(frame_counts > 0)
        if active_pixels.size == 0:
            continue
        for row, col in active_pixels:
            count = int(frame_counts[row, col])
            if count <= 0:
                continue
            times = frame_idx * dt + np.sort(rng.uniform(0.0, dt, size=count))
            if timing_jitter_ns > 0 and times.size > 0:
                times = times + rng.normal(0.0, timing_jitter_ns * 1e-9, size=times.size)
            if tdc_bin_width_ns > 0 and times.size > 0:
                bin_s = tdc_bin_width_ns * 1e-9
                times = np.round(times / bin_s) * bin_s
            upper_bound = float(
                np.nextafter(np.float32((frame_idx + 1) * dt), np.float32(frame_idx * dt))
            )
            times = np.clip(times, frame_idx * dt, upper_bound)
            event_times.append(times.astype(np.float32))
            event_pixels.append(np.full(times.size, row * roi_w + col, dtype=np.int32))

    if not event_times:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)
    return np.concatenate(event_times), np.concatenate(event_pixels)


def _pixel_events_from_counts(counts: np.ndarray) -> np.ndarray:
    flat = counts.reshape(counts.shape[0], -1)
    frame_ids = []
    pixel_ids = []
    for frame_idx in range(flat.shape[0]):
        nonzero = np.nonzero(flat[frame_idx])[0]
        for pixel_idx in nonzero:
            pixel_ids.extend([pixel_idx] * int(flat[frame_idx, pixel_idx]))
            frame_ids.extend([frame_idx] * int(flat[frame_idx, pixel_idx]))
    if not pixel_ids:
        return np.zeros((0, 2), dtype=np.int32)
    return np.column_stack([np.asarray(frame_ids, dtype=np.int32), np.asarray(pixel_ids, dtype=np.int32)])


def _point_in_polygon(col: float, row: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > row) != (yj > row)) and (
            col < (xj - xi) * (row - yi) / max(yj - yi, 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for i, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(i + 1) % len(polygon)]
        area += x0 * y1 - x1 * y0
    return abs(area) * 0.5


def _add_projected_polygon(mask: np.ndarray, polygon: list[tuple[float, float]], weight: float) -> None:
    if len(polygon) < 3 or weight <= 0:
        return
    roi_h, roi_w = mask.shape
    area = _polygon_area(polygon)
    if area < 0.5:
        col = int(np.floor(np.mean([p[0] for p in polygon])))
        row = int(np.floor(np.mean([p[1] for p in polygon])))
        if 0 <= row < roi_h and 0 <= col < roi_w:
            mask[row, col] += max(area, 1e-3) * weight
        return

    min_col = max(0, int(np.floor(min(p[0] for p in polygon))))
    max_col = min(roi_w - 1, int(np.ceil(max(p[0] for p in polygon))))
    min_row = max(0, int(np.floor(min(p[1] for p in polygon))))
    max_row = min(roi_h - 1, int(np.ceil(max(p[1] for p in polygon))))
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            if _point_in_polygon(col + 0.5, row + 0.5, polygon):
                mask[row, col] += weight


def _project_blade_local_points(
    cx: float,
    cy: float,
    local_points: list[tuple[float, float]],
    angle_rad: float,
    pitch_rad: float,
    meters_to_pixels: float,
) -> list[tuple[float, float]]:
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    sin_p = float(np.sin(pitch_rad))
    projected: list[tuple[float, float]] = []
    for x_m, z_m in local_points:
        yawed_x = x_m * cos_a - z_m * sin_a
        yawed_z = x_m * sin_a + z_m * cos_a
        final_y = -yawed_z * sin_p
        projected.append((cx + yawed_x * meters_to_pixels, cy - final_y * meters_to_pixels))
    return projected


def _series_value(value, frame_idx: int, fallback: float) -> float:
    if value is None:
        return float(fallback)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return float(fallback)
    idx = min(frame_idx, arr.size - 1)
    val = float(arr.reshape(-1)[idx])
    return val if np.isfinite(val) else float(fallback)


def _matrix_series_value(value, frame_idx: int, col_idx: int, fallback: float) -> float:
    if value is None:
        return float(fallback)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return float(fallback)
    if arr.ndim == 1:
        return _series_value(arr, frame_idx, fallback)
    row_idx = min(frame_idx, arr.shape[0] - 1)
    col_idx = min(col_idx, arr.shape[1] - 1)
    val = float(arr[row_idx, col_idx])
    return val if np.isfinite(val) else float(fallback)


def _has_external_target(params: SimParams) -> bool:
    return (
        getattr(params.geometry, "external_target_x_m", None) is not None
        and getattr(params.geometry, "external_target_y_m", None) is not None
        and getattr(params.geometry, "external_target_z_m", None) is not None
    )


def _external_target_position(params: SimParams, frame_idx: int) -> tuple[float, float, float] | None:
    if not _has_external_target(params):
        return None
    return (
        _series_value(params.geometry.external_target_x_m, frame_idx, 0.0),
        _series_value(params.geometry.external_target_y_m, frame_idx, 0.0),
        _series_value(params.geometry.external_target_z_m, frame_idx, 0.0),
    )


def _project_world_point_to_pixel(
    point: tuple[float, float, float],
    params: SimParams,
    roi_h: int,
    roi_w: int,
) -> tuple[float, float, float] | None:
    """Replicate the frontend ordinary simulation camera projection."""

    detector_x = float(getattr(params.geometry, "detector_position_x_m", 0.0))
    detector_y = float(getattr(params.geometry, "detector_position_y_m", 0.0))
    detector_z = float(getattr(params.geometry, "detector_position_z_m", 0.0))
    rel_x = float(point[0]) - detector_x
    rel_y = float(point[1]) - detector_y
    rel_z = float(point[2]) - detector_z

    yaw_rad = np.deg2rad(float(getattr(params.geometry, "detector_yaw_deg", 0.0)))
    pitch_rad = np.deg2rad(float(getattr(params.geometry, "detector_pitch_deg", 0.0)))
    cos_yaw = float(np.cos(-yaw_rad))
    sin_yaw = float(np.sin(-yaw_rad))
    yawed_x = rel_x * cos_yaw - rel_z * sin_yaw
    yawed_z = rel_x * sin_yaw + rel_z * cos_yaw

    cos_pitch = float(np.cos(-pitch_rad))
    sin_pitch = float(np.sin(-pitch_rad))
    local_y = rel_y * cos_pitch - yawed_z * sin_pitch
    local_z = rel_y * sin_pitch + yawed_z * cos_pitch
    if local_z <= 0.1:
        return None

    fov_rad = max(float(params.optical.detector_fov_urad) * 1e-6, 1e-9)
    f_pixel = (roi_w / 2.0) / max(float(np.tan(fov_rad / 2.0)), 1e-12)
    col = roi_w / 2.0 + f_pixel * (yawed_x / local_z)
    row = roi_h / 2.0 - f_pixel * (local_y / local_z)
    distance = float(np.sqrt(yawed_x * yawed_x + local_y * local_y + local_z * local_z))
    return col, row, distance


def _frontend_blade_world_points(
    target_pos: tuple[float, float, float],
    local_points: list[tuple[float, float]],
    angle_rad: float,
    pitch_rad: float,
) -> list[tuple[float, float, float]]:
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    cos_p = float(np.cos(pitch_rad))
    sin_p = float(np.sin(pitch_rad))
    points: list[tuple[float, float, float]] = []
    for x_m, z_m in local_points:
        yawed_x = x_m * cos_a - z_m * sin_a
        yawed_z = x_m * sin_a + z_m * cos_a
        points.append((
            target_pos[0] + yawed_x,
            target_pos[1] - yawed_z * sin_p,
            target_pos[2] + yawed_z * cos_p,
        ))
    return points


def _frontend_drone_world_points(
    target_pos: tuple[float, float, float],
    local_points: list[tuple[float, float]],
    pitch_rad: float,
    yaw_rad: float,
    roll_rad: float,
) -> list[tuple[float, float, float]]:
    cos_p = float(np.cos(pitch_rad))
    sin_p = float(np.sin(pitch_rad))
    cos_y = float(np.cos(yaw_rad))
    sin_y = float(np.sin(yaw_rad))
    cos_r = float(np.cos(roll_rad))
    sin_r = float(np.sin(roll_rad))
    points: list[tuple[float, float, float]] = []
    for x_m, z_m in local_points:
        pitched_y = -z_m * sin_p
        pitched_z = z_m * cos_p
        rolled_x = x_m * cos_r - pitched_y * sin_r
        rolled_y = x_m * sin_r + pitched_y * cos_r
        yawed_x = rolled_x * cos_y + pitched_z * sin_y
        yawed_z = -rolled_x * sin_y + pitched_z * cos_y
        points.append((target_pos[0] + yawed_x, target_pos[1] + rolled_y, target_pos[2] + yawed_z))
    return points


def _project_world_polygon(
    points: list[tuple[float, float, float]],
    params: SimParams,
    roi_h: int,
    roi_w: int,
) -> list[tuple[float, float]]:
    projected: list[tuple[float, float]] = []
    for point in points:
        pixel = _project_world_point_to_pixel(point, params, roi_h, roi_w)
        if pixel is None:
            continue
        col, row, _distance = pixel
        projected.append((col, row))
    return projected


def _shape_signal_distribution_cube(
    signal_total_t: np.ndarray,
    cx_t: np.ndarray,
    cy_t: np.ndarray,
    roi_h: int,
    roi_w: int,
    width_px_t: np.ndarray,
    height_px_t: np.ndarray,
    t: np.ndarray,
    params: SimParams,
    blur_sigma_x: float,
    blur_sigma_y: float,
    pde_map: np.ndarray,
) -> np.ndarray:
    shape = str(params.target.body_shape or "")
    if shape not in {"sphere", "blade_strip", "drone_quad"}:
        return signal_distribution_cube_projected(
            signal_total_t,
            cx_t,
            cy_t,
            roi_h,
            roi_w,
            width_px_t=width_px_t,
            height_px_t=height_px_t,
            blur_sigma_x=blur_sigma_x,
            blur_sigma_y=blur_sigma_y,
            pde_map=pde_map,
        )

    yy, xx = np.mgrid[0:roi_h, 0:roi_w].astype(np.float64)
    cube = np.zeros((signal_total_t.size, roi_h, roi_w), dtype=np.float64)
    use_frontend_projection = _has_external_target(params)
    shape_mask_cache: dict[tuple, np.ndarray] = {}

    for frame_idx in range(signal_total_t.size):
        photons = float(signal_total_t[frame_idx])
        if photons <= 0:
            continue

        cx = float(cx_t[frame_idx])
        cy = float(cy_t[frame_idx])
        width = float(max(width_px_t[frame_idx], 1e-6))
        height = float(max(height_px_t[frame_idx], 1e-6))

        target_pos = _external_target_position(params, frame_idx) if use_frontend_projection else None

        if shape == "sphere":
            if target_pos is not None:
                center = _project_world_point_to_pixel(target_pos, params, roi_h, roi_w)
                if center is None:
                    continue
                cx, cy, distance = center
                fov_rad = max(float(params.optical.detector_fov_urad) * 1e-6, 1e-9)
                f_pixel = (roi_w / 2.0) / max(float(np.tan(fov_rad / 2.0)), 1e-12)
                radius = max(0.35, f_pixel * (float(params.target.target_width_m) * 0.5 / max(distance, 1e-9)))
            else:
                radius = max(0.35, 0.5 * max(width, height))
            rr = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2
            mask = np.where(rr <= radius**2, np.sqrt(np.clip(1.0 - rr / max(radius**2, 1e-12), 0.0, 1.0)), 0.0)
            if np.sum(mask) <= 0:
                sigma = max(0.45, radius * 0.5)
                mask = np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))

        elif shape == "blade_strip":
            spin_hz = max(_series_value(getattr(params.geometry, "external_spin_hz", None), frame_idx, params.target.spin_hz), 0.0)
            phase = _series_value(getattr(params.geometry, "external_rotation_phase_rad", None), frame_idx, np.nan)
            angle = phase if np.isfinite(phase) else 2.0 * np.pi * spin_hz * float(t[frame_idx]) + float(params.target.phase1)
            pitch_deg = _series_value(getattr(params.geometry, "external_pitch_deg", None), frame_idx, getattr(params.target, "orientation_pitch_deg", 0.0))
            pitch_rad = np.deg2rad(pitch_deg)
            meters_to_pixels = max(width, height, 1e-6) / max(float(params.target.target_length_m), 1e-6)
            custom_x = getattr(params.target, "custom_shape_x", None)
            custom_y = getattr(params.target, "custom_shape_y", None)
            custom_i = getattr(params.target, "custom_shape_intensity", None)
            if custom_x is not None and custom_y is not None and custom_i is not None and len(custom_x) == len(custom_y) == len(custom_i):
                cache_key = (
                    "custom_blade",
                    int(round((float(angle) % (2.0 * np.pi)) / (2.0 * np.pi) * 128.0)) % 128,
                    round(float(pitch_rad), 6),
                    tuple(round(float(v), 5) for v in target_pos) if target_pos is not None else None,
                    round(float(cx), 4),
                    round(float(cy), 4),
                    round(float(meters_to_pixels), 4),
                )
                cached_mask = shape_mask_cache.get(cache_key)
                if cached_mask is not None:
                    mask = cached_mask
                else:
                    mask = np.zeros((roi_h, roi_w), dtype=np.float64)
                    aspect = max(float(getattr(params.target, "custom_shape_aspect_ratio", 1.0)), 1e-6)
                    blade_length = float(params.target.target_length_m)
                    geom_w = blade_length if aspect >= 1.0 else blade_length * aspect
                    geom_h = blade_length / aspect if aspect >= 1.0 else blade_length
                    for px_n, py_n, intensity in zip(custom_x, custom_y, custom_i):
                        local_point = (float(px_n) * geom_w, float(py_n) * geom_h)
                        if target_pos is not None:
                            world_point = _frontend_blade_world_points(target_pos, [local_point], angle, pitch_rad)[0]
                            projected = _project_world_point_to_pixel(world_point, params, roi_h, roi_w)
                            if projected is None:
                                continue
                            col_f, row_f, _distance = projected
                        else:
                            [(col_f, row_f)] = _project_blade_local_points(
                                cx,
                                cy,
                                [local_point],
                                angle,
                                pitch_rad,
                                meters_to_pixels,
                            )
                        col = int(round(col_f))
                        row = int(round(row_f))
                        if 0 <= row < roi_h and 0 <= col < roi_w:
                            mask[row, col] += max(float(intensity), 0.0)
                    if np.sum(mask) <= 0 and target_pos is None:
                        mask[int(np.clip(round(cy), 0, roi_h - 1)), int(np.clip(round(cx), 0, roi_w - 1))] = 1.0
                    if len(shape_mask_cache) < 256:
                        shape_mask_cache[cache_key] = mask
            else:
                mask = np.zeros((roi_h, roi_w), dtype=np.float64)
                blade_length = float(max(params.target.target_length_m, 1e-6))
                blade_width = float(max(params.target.target_width_m, 1e-6))
                corners = [
                    (-blade_width / 2.0, 0.0),
                    (blade_width / 2.0, 0.0),
                    (blade_width / 2.0, blade_length),
                    (-blade_width / 2.0, blade_length),
                ]
                if target_pos is not None:
                    world_points = _frontend_blade_world_points(target_pos, corners, angle, pitch_rad)
                    projected = _project_world_polygon(world_points, params, roi_h, roi_w)
                else:
                    projected = _project_blade_local_points(cx, cy, corners, angle, pitch_rad, meters_to_pixels)
                _add_projected_polygon(mask, projected, 1.0)

        else:
            mask = np.zeros((roi_h, roi_w), dtype=np.float64)
            drone_width = float(max(params.target.target_width_m, 1e-6))
            drone_length = float(max(params.target.target_length_m, 1e-6))
            meters_to_pixels = max(width / drone_width, height / drone_length, 1e-6)
            pitch_rad = np.deg2rad(_series_value(getattr(params.geometry, "external_pitch_deg", None), frame_idx, getattr(params.target, "orientation_pitch_deg", 0.0)))
            yaw_rad = np.deg2rad(_series_value(getattr(params.geometry, "external_yaw_deg", None), frame_idx, getattr(params.target, "orientation_yaw_deg", 0.0)))
            roll_rad = np.deg2rad(_series_value(getattr(params.geometry, "external_roll_deg", None), frame_idx, getattr(params.target, "orientation_roll_deg", 0.0)))
            cos_y = float(np.cos(yaw_rad))
            sin_y = float(np.sin(yaw_rad))
            cos_r = float(np.cos(roll_rad))
            sin_r = float(np.sin(roll_rad))

            def project_drone_point(x_m: float, z_m: float) -> tuple[float, float]:
                pitched_y = -z_m * float(np.sin(pitch_rad))
                pitched_z = z_m * float(np.cos(pitch_rad))
                rolled_x = x_m * cos_r - pitched_y * sin_r
                rolled_y = x_m * sin_r + pitched_y * cos_r
                yawed_x = rolled_x * cos_y + pitched_z * sin_y
                return cx + yawed_x * meters_to_pixels, cy - rolled_y * meters_to_pixels

            body_half_w = drone_width * 0.19
            body_half_l = drone_length * 0.21
            body = [
                (-body_half_w, -body_half_l),
                (body_half_w, -body_half_l),
                (body_half_w, body_half_l),
                (-body_half_w, body_half_l),
            ]
            if target_pos is not None:
                _add_projected_polygon(
                    mask,
                    _project_world_polygon(
                        _frontend_drone_world_points(target_pos, body, pitch_rad, yaw_rad, roll_rad),
                        params,
                        roi_h,
                        roi_w,
                    ),
                    0.9,
                )
            else:
                _add_projected_polygon(mask, [project_drone_point(x, z) for x, z in body], 0.9)

            prop_width = max(0.01, float(params.target.propeller_diameter_m) * 0.08)
            prop_len = float(params.target.propeller_diameter_m)
            arm_half_w = drone_width * 0.36
            arm_half_l = drone_length * 0.36
            prop_centers = [
                (arm_half_w, arm_half_l, 1),
                (-arm_half_w, arm_half_l, 2),
                (arm_half_w, -arm_half_l, 3),
                (-arm_half_w, -arm_half_l, 4),
            ]
            for pc_x, pc_z, prop_id in prop_centers:
                direction = 1.0 if prop_id in (1, 4) else -1.0
                prop_col = int(prop_id) - 1
                prop_spin_hz = max(
                    _matrix_series_value(
                        getattr(params.geometry, "external_propeller_spin_hz", None),
                        frame_idx,
                        prop_col,
                        _series_value(getattr(params.geometry, "external_spin_hz", None), frame_idx, params.target.spin_hz),
                    ),
                    0.0,
                )
                phase_fallback = 2.0 * np.pi * prop_spin_hz * float(t[frame_idx])
                prop_phase = _matrix_series_value(
                    getattr(params.geometry, "external_propeller_phase_rad", None),
                    frame_idx,
                    prop_col,
                    phase_fallback,
                )
                prop_angle = direction * prop_phase
                half_l = prop_len / 2.0
                half_w = prop_width / 2.0
                if prop_spin_hz > 1.0:
                    local = [
                        (
                            pc_x + half_l * float(np.cos(theta)),
                            pc_z + half_l * float(np.sin(theta)),
                        )
                        for theta in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
                    ]
                else:
                    cos_a = float(np.cos(prop_angle))
                    sin_a = float(np.sin(prop_angle))
                    blade_corners = [(-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)]
                    local = [
                        (lx * cos_a - lz * sin_a + pc_x, lx * sin_a + lz * cos_a + pc_z)
                        for lx, lz in blade_corners
                    ]
                prop_poly = []
                for x_m, z_m in local:
                    prop_poly.append(project_drone_point(x_m, z_m))
                if target_pos is not None:
                    prop_poly = _project_world_polygon(
                        _frontend_drone_world_points(target_pos, local, pitch_rad, yaw_rad, roll_rad),
                        params,
                        roi_h,
                        roi_w,
                    )
                _add_projected_polygon(mask, prop_poly, max(float(params.target.propeller_reflectivity), 0.0))

        peak = float(np.max(mask))
        if peak > 0:
            mask = np.where(mask >= peak * 1e-3, mask, 0.0)
        total = float(np.sum(mask))
        if total <= 0:
            continue
        cube[frame_idx] = photons * mask / total

    totals = np.sum(cube, axis=(1, 2), keepdims=True)
    cube = np.divide(cube, np.maximum(totals, 1e-12), out=np.zeros_like(cube), where=totals > 0) * signal_total_t[:, None, None]
    display_blur_x = max(float(blur_sigma_x), 1.05 if str(params.target.body_shape or "") == "drone_quad" else 0.65)
    display_blur_y = max(float(blur_sigma_y), 1.05 if str(params.target.body_shape or "") == "drone_quad" else 0.65)
    cube = _apply_separable_blur(cube, display_blur_x, display_blur_y)
    totals = np.sum(cube, axis=(1, 2), keepdims=True)
    cube = np.divide(cube, np.maximum(totals, 1e-12), out=np.zeros_like(cube), where=totals > 0) * signal_total_t[:, None, None]
    return cube * pde_map[None, :, :]


def simulate_active_spad(params: SimParams):
    rng = np.random.default_rng(params.seed)
    t, dt, n_frames = make_time_axis(params.observation_time_s, params.sample_rate_hz)
    roi_h = params.image.roi_h
    roi_w = params.image.roi_w

    R_t, sun_unit_t, los_unit_t, phase_angle_t, assumptions = _build_geometry(t, params)
    (
        signal_scale_t,
        target_rate_cps_t,
        solar_rate_cps_t,
        laser_rate_cps_t,
        truth_freq_hz,
        truth_precession_hz,
        harmonic_truth_strength,
        projection_rotation_t,
    ) = _target_signal(
        t, R_t, sun_unit_t, los_unit_t, phase_angle_t, params
    )
    total_fov_urad, pixel_ifov_urad, off_axis_urad = _detector_geometry(params)
    fov_half_urad = total_fov_urad / 2.0
    sigma_scale = np.clip(50.0 / max(pixel_ifov_urad, 1e-6), 0.65, 2.5)

    visibility_t = make_visibility_mask(
        t,
        params.target.outage_fraction,
        seed=params.target.outage_seed,
        mode=getattr(params.target, "outage_mode", "random_segments"),
        period_s=getattr(params.target, "outage_period_s", 6.0),
        on_fraction=getattr(params.target, "outage_on_fraction", 0.33),
        gap_start_fraction=getattr(params.target, "outage_gap_start_fraction", 0.33),
        gap_end_fraction=getattr(params.target, "outage_gap_end_fraction", 0.66),
    ).astype(np.float64)
    if getattr(params.geometry, "trajectory_name", "") == "manual_recorded_flight":
        external_visibility = _external_series(
            params.geometry.external_visibility,
            n_frames,
            "external_visibility",
        )
        visibility_t = visibility_t * np.clip(external_visibility, 0.0, 1.0)
    glint_t = np.ones(n_frames, dtype=np.float64)
    if params.target.glint_probability > 0:
        glint_flags = rng.random(n_frames) < params.target.glint_probability
        glint_t[glint_flags] = params.target.glint_gain

    bg_base_t, bg_scene_stray_t = _background_series(t, phase_angle_t, los_unit_t, params)
    bg_scale = _background_scaling(params, pixel_ifov_urad)
    bg_base_t = bg_base_t * bg_scale
    bg_scene_stray_t = bg_scene_stray_t * bg_scale
    bg_spatial = background_spatial_map(
        roi_h,
        roi_w,
        sigma=params.background.spatial_nonuniformity_sigma,
        gradient_x=params.background.gradient_x,
        gradient_y=params.background.gradient_y,
        rng=rng,
    )
    bg_cube = bg_base_t[:, None, None] * bg_spatial[None, :, :]

    pde_map = make_pde_map(
        rng,
        roi_h,
        roi_w,
        sigma=params.spad.pde_nonuniform_sigma,
        hot_pixel_fraction=params.spad.hot_pixel_fraction,
        hot_pixel_scale=params.spad.hot_pixel_scale,
    )
    dark_map = make_dark_map(
        rng,
        roi_h,
        roi_w,
        base_rate_cps=params.spad.dark_count_rate_cps,
        sigma_frac=params.spad.dark_count_sigma,
        hot_pixel_fraction=params.spad.hot_pixel_fraction,
        hot_pixel_scale=params.spad.hot_pixel_scale,
    )

    blur_sigma_x = (
        params.image.spot_sigma_x_pixels if params.image.spot_sigma_x_pixels > 0 else params.image.spot_sigma_pixels
    )
    blur_sigma_y = params.image.spot_sigma_y_pixels if params.image.spot_sigma_y_pixels > 0 else blur_sigma_x
    if params.image.jitter_model == "gauss_markov":
        jx, jy = jitter_series_gauss_markov(rng, n_frames, params.image.jitter_sigma_pixels, params.image.jitter_corr_time_s, dt)
    elif params.image.jitter_model == "sinusoidal":
        jx, jy = jitter_series_sinusoidal(t, params.image.jitter_sigma_pixels, freq_hz=max(truth_freq_hz * 0.1, 0.2))
    else:
        jx, jy = jitter_series_white(rng, n_frames, params.image.jitter_sigma_pixels)

    scene_mode = getattr(params.image, "scene_mode", "centered_roi")
    trajectory_name = getattr(params.geometry, "trajectory_name", "")
    if scene_mode == "centered_roi":
        cx_t, cy_t = target_centroid_series(
            t,
            params.image.center_x,
            params.image.center_y,
            params.image.drift_pixels_per_s_x,
            params.image.drift_pixels_per_s_y,
            jx,
            jy,
        )
        off_axis_x_urad_t = (cx_t - params.image.center_x) * pixel_ifov_urad
        off_axis_y_urad_t = (cy_t - params.image.center_y) * pixel_ifov_urad
        truth_in_fov_t = (
            np.hypot(off_axis_x_urad_t, off_axis_y_urad_t) <= fov_half_urad
        ).astype(np.float32)
        truth_reacquire_t = np.zeros(n_frames, dtype=np.float32)
    else:
        default_center_x = params.image.center_x
        default_center_y = params.image.center_y
        initial_center_x = params.image.initial_center_x if params.image.initial_center_x >= 0 else default_center_x
        initial_center_y = params.image.initial_center_y if params.image.initial_center_y >= 0 else default_center_y
        initial_x_urad = (initial_center_x - default_center_x) * pixel_ifov_urad
        initial_y_urad = (initial_center_y - default_center_y) * pixel_ifov_urad
        if trajectory_name == "manual_recorded_flight":
            off_axis_x_urad_t = _external_series(
                params.geometry.external_off_axis_x_urad,
                n_frames,
                "external_off_axis_x_urad",
            )
            off_axis_y_urad_t = _external_series(
                params.geometry.external_off_axis_y_urad,
                n_frames,
                "external_off_axis_y_urad",
            )
            truth_in_fov_t = np.asarray(
                _external_series(params.geometry.external_in_fov, n_frames, "external_in_fov"),
                dtype=np.float32,
            )
            truth_reacquire_t = np.zeros(n_frames, dtype=np.float32)
        else:
            off_axis_x_urad_t, off_axis_y_urad_t, truth_in_fov_t, truth_reacquire_t = boresight_offset_series(
                t,
                scene_mode,
                initial_x_urad=initial_x_urad,
                initial_y_urad=initial_y_urad,
                angular_rate_urad_s_x=getattr(params.geometry, "angular_rate_urad_s_x", 0.0),
                angular_rate_urad_s_y=getattr(params.geometry, "angular_rate_urad_s_y", 0.0),
                tracking_residual_sigma_urad=getattr(params.geometry, "tracking_residual_sigma_urad", 0.0),
                trajectory_name=getattr(params.geometry, "trajectory_name", ""),
                reacquire_enabled=getattr(params.geometry, "reacquire_enabled", False),
                reacquire_start_fraction=getattr(params.geometry, "reacquire_start_fraction", 0.35),
                reacquire_end_fraction=getattr(params.geometry, "reacquire_end_fraction", 0.55),
                fov_urad=total_fov_urad,
                edge_entry_mode=getattr(params.image, "edge_entry_mode", "manual"),
                rng=rng,
            )
        cx_t, cy_t = centroid_from_off_axis_series(
            off_axis_x_urad_t,
            off_axis_y_urad_t,
            roi_w,
            roi_h,
            pixel_ifov_urad,
            params.image.center_x,
            params.image.center_y,
            drift_x=params.image.drift_pixels_per_s_x,
            drift_y=params.image.drift_pixels_per_s_y,
            t=t,
            jitter_x=jx,
            jitter_y=jy,
        )

    radial_off_axis_t = np.hypot(off_axis_x_urad_t, off_axis_y_urad_t)
    fov_visibility_t = np.where(
        radial_off_axis_t >= fov_half_urad,
        0.0,
        np.maximum(0.0, 1.0 - (radial_off_axis_t / max(fov_half_urad, 1e-6)) ** 2),
    )
    fov_visibility_t = np.minimum(fov_visibility_t, truth_in_fov_t.astype(np.float64))
    edge_margin_t = np.clip((fov_half_urad - radial_off_axis_t) / max(fov_half_urad, 1e-6), 0.0, 1.0)
    fov_visibility = float(np.mean(fov_visibility_t))
    edge_margin = max(0.35, float(np.mean(edge_margin_t[truth_in_fov_t > 0])) if np.any(truth_in_fov_t > 0) else 0.35)
    blur_sigma_x = max(0.05, blur_sigma_x * sigma_scale / max(edge_margin, 0.35))
    blur_sigma_y = max(0.05, blur_sigma_y * sigma_scale / max(edge_margin, 0.35))
    atmospheric_path_length_m = np.minimum(
        R_t,
        max(float(getattr(params.optical, "atmospheric_path_length_m", 6_000.0)), 0.0),
    )
    atmospheric_t = atmospheric_transmittance(
        wavelength_nm=params.optical.wavelength_nm,
        visibility_km=params.optical.atmospheric_visibility_km,
        distance_m=atmospheric_path_length_m,
        enabled=params.optical.atmospheric_attenuation_enabled,
    )
    atmospheric_transmission_mean = float(np.mean(atmospheric_t))
    common_detection_factor_t = atmospheric_t * visibility_t * glint_t * fov_visibility_t
    final_solar_rate_cps_t = solar_rate_cps_t * common_detection_factor_t
    final_laser_rate_cps_t = laser_rate_cps_t * common_detection_factor_t
    final_rate_cps_t = final_solar_rate_cps_t + final_laser_rate_cps_t
    expected_signal_t = final_rate_cps_t * dt
    body_vertices = simple_body_vertices(params.target.body_shape, target_area_m2=params.target.target_area_m2)
    projected_width_urad_t, projected_height_urad_t = projected_extent_series_from_vertices(
        body_vertices,
        projection_rotation_t,
        los_unit_t,
        R_t,
    )
    projected_width_px_t = projected_width_urad_t / max(pixel_ifov_urad, 1e-6)
    projected_height_px_t = projected_height_urad_t / max(pixel_ifov_urad, 1e-6)
    signal_cube = _shape_signal_distribution_cube(
        expected_signal_t,
        cx_t,
        cy_t,
        roi_h,
        roi_w,
        projected_width_px_t,
        projected_height_px_t,
        t,
        params,
        blur_sigma_x,
        blur_sigma_y,
        pde_map,
    )
    pde_safe = np.maximum(pde_map, 1e-12)
    expected_signal_map = np.sum(signal_cube / pde_safe[None, :, :], axis=0).astype(np.float32)

    dark_expected_cube = dark_map[None, :, :] * dt
    bg_expected_cube = bg_cube * dt * pde_map[None, :, :]
    mu_total = signal_cube + bg_expected_cube + dark_expected_cube
    ideal_rate = mu_total / max(dt, 1e-12)
    effective_rate = apply_dead_time_rate(ideal_rate, params.spad.dead_time_ns * 1e-9, model=params.spad.dead_time_model)
    mu_corrected = effective_rate * dt
    counts, sample_backend = sample_poisson_counts_accelerated(
        mu_corrected,
        rng,
        requested_backend=params.compute_backend,
        seed=params.seed,
    )

    if params.spad.afterpulse_probability > 0:
        afterpulse = rng.poisson(mu_corrected * params.spad.afterpulse_probability)
        counts = counts + afterpulse
    if params.spad.crosstalk_probability > 0:
        shifted = np.zeros_like(counts)
        shifted[:, :, 1:] = counts[:, :, :-1]
        counts = counts + rng.poisson(shifted * params.spad.crosstalk_probability)

    saturation_mask = counts > params.spad.max_count_per_frame
    counts = np.clip(counts, 0, params.spad.max_count_per_frame).astype(np.uint16)

    mean_signal_per_frame = float(np.mean(np.sum(signal_cube, axis=(1, 2))))
    mean_background_per_frame = float(np.mean(np.sum(bg_expected_cube, axis=(1, 2))))
    mean_dark_per_frame = float(np.mean(np.sum(dark_expected_cube, axis=(1, 2))))
    total_signal = float(np.sum(signal_cube))
    total_background = float(np.sum(bg_expected_cube))
    total_noise = float(np.sum(bg_expected_cube + dark_expected_cube))
    dead_time_loss_ratio = float(
        np.clip(
            1.0 - (np.sum(mu_corrected) + 1e-12) / (np.sum(mu_total) + 1e-12),
            0.0,
            1.0,
        )
    )
    snr_db = _shot_noise_snr_db(total_signal, total_noise)

    valid_truth = truth_in_fov_t > 0
    truth_row = int(np.clip(round(float(np.mean(cy_t[valid_truth])) if np.any(valid_truth) else float(np.mean(cy_t))), 0, roi_h - 1))
    truth_col = int(np.clip(round(float(np.mean(cx_t[valid_truth])) if np.any(valid_truth) else float(np.mean(cx_t))), 0, roi_w - 1))
    truth_pixel = truth_row * roi_w + truth_col

    event_times = None
    event_pixels = None
    if params.simulation_mode == "event" or params.save_event_list:
        max_events = 5_000_000
        event_count = int(np.sum(counts, dtype=np.int64))
        if event_count > max_events:
            raise ValueError(
                f"event output budget exceeded: sampled {event_count} events > {max_events}"
            )
        event_times, event_pixels = _event_stream_from_counts(
            counts,
            dt,
            params.spad.timing_jitter_ns,
            params.spad.tdc_bin_width_ns,
            rng,
        )

    assumptions = list(dict.fromkeys(assumptions))
    warnings: list[str] = []
    if params.simulation_tier == "physics_informed":
        warnings.append("Physics-informed mode is geometry-aware but uses a near-range scene model, not full wave-optics propagation.")
    if params.simulation_mode == "event":
        warnings.append("Event timestamps are synthesized from dead-time-corrected frame counts with timing jitter and TDC quantization, not full TCSPC transport.")
    if params.sample_rate_hz >= 1e4:
        warnings.append("High sample rate sets narrow time bins; low counts per bin can still be valid if total photon budget remains sufficient.")
    if fov_visibility < 0.999:
        warnings.append("Target is partially off boresight and signal is clipped by the detector field of view.")
    if scene_mode != "centered_roi":
        assumptions.append("Scene mode uses geometry-informed focal-plane motion instead of a fixed centered ROI target.")
    assumptions.append("Focal-plane signal uses a projected-silhouette footprint with only a light optical blur layer.")
    assumptions.append("Reported SNR uses the aggregate Poisson shot-noise approximation S/sqrt(S+B+D).")
    if params.illumination.mode in {"laser", "laser_plus_solar"}:
        assumptions.append("Laser return uses an on-axis Gaussian-beam engineering approximation with an independent transmitter divergence and capped intercepted power.")
    if params.illumination.mode in {"solar", "laser_plus_solar"}:
        assumptions.append("Solar-reflection signal is computed independently from solar-driven scene stray photons.")

    preset = preset_summary(params)
    detector_summary = {
        "preset": params.detector_preset,
        "preset_label": preset.label,
        "array_rows": roi_h,
        "array_cols": roi_w,
        "pixel_pitch_um": params.image.pixel_pitch_um,
        "fill_factor": params.image.fill_factor,
        "microlens_gain": params.image.microlens_gain,
        "detector_fov_urad": total_fov_urad,
        "pixel_ifov_urad": pixel_ifov_urad,
        "tdc_bin_width_ns": params.spad.tdc_bin_width_ns,
        "irf_fwhm_ps": params.spad.irf_fwhm_ps,
        "max_count_rate_cps_per_pixel": params.spad.max_count_rate_cps_per_pixel,
        "max_count_per_frame": params.spad.max_count_per_frame,
        "quantum_efficiency": params.optical.quantum_efficiency,
        "receiver_efficiency": params.optical.receiver_efficiency,
        "assumptions": preset.assumptions,
    }

    result = {
        "scenario_id": None,
        "simulation_tier": params.simulation_tier,
        "sample_backend": sample_backend,
        "output_mode": params.simulation_mode,
        "scene_mode": scene_mode,
        "lightcurve_mode": "attitude_driven"
        if (params.lightcurve_mode == "attitude_driven" or (params.simulation_tier == "physics_informed" and params.target.spin_hz > 0))
        else "analytic_modulation",
        "assumptions": assumptions,
        "warnings": warnings,
        "counts": counts,
        "expected_signal_map": expected_signal_map,
        "event_times": event_times,
        "event_pixels": event_pixels,
        "n_frames": n_frames,
        "roi_h": roi_h,
        "roi_w": roi_w,
        "sample_rate_hz": params.sample_rate_hz,
        "truth_freq_hz": truth_freq_hz,
        "truth_precession_hz": truth_precession_hz,
        "truth_pixel": truth_pixel,
        "truth_row": truth_row,
        "truth_col": truth_col,
        "truth_cx_series": cx_t.astype(np.float32) if params.save_truth_series else None,
        "truth_cy_series": cy_t.astype(np.float32) if params.save_truth_series else None,
        "truth_signal_series": expected_signal_t.astype(np.float32) if params.save_truth_series else None,
        "truth_bg_base_series": bg_base_t.astype(np.float32) if params.save_truth_series else None,
        "truth_bg_total_series": np.sum(bg_expected_cube, axis=(1, 2)).astype(np.float32) if params.save_truth_series else None,
        "truth_visibility_series": visibility_t.astype(np.float32) if params.save_truth_series else None,
        "truth_in_fov_series": truth_in_fov_t.astype(np.float32) if params.save_truth_series else None,
        "truth_reacquire_flag_series": truth_reacquire_t.astype(np.float32) if params.save_truth_series else None,
        "truth_off_axis_x_urad_series": off_axis_x_urad_t.astype(np.float32) if params.save_truth_series else None,
        "truth_off_axis_y_urad_series": off_axis_y_urad_t.astype(np.float32) if params.save_truth_series else None,
        "truth_glint_series": glint_t.astype(np.float32) if params.save_truth_series else None,
        "truth_range_series": R_t.astype(np.float32) if params.save_truth_series else None,
        "truth_phase_angle_series": phase_angle_t.astype(np.float32) if params.save_truth_series else None,
        "truth_signal_scale_series": signal_scale_t.astype(np.float32) if params.save_truth_series else None,
        "truth_rate_cps_series": final_rate_cps_t.astype(np.float32) if params.save_truth_series else None,
        "truth_projected_width_px_series": projected_width_px_t.astype(np.float32) if params.save_truth_series else None,
        "truth_projected_height_px_series": projected_height_px_t.astype(np.float32) if params.save_truth_series else None,
        "truth_projected_width_urad_series": projected_width_urad_t.astype(np.float32) if params.save_truth_series else None,
        "truth_projected_height_urad_series": projected_height_urad_t.astype(np.float32) if params.save_truth_series else None,
        "modulation_series": signal_scale_t.astype(np.float32) if params.save_truth_series else None,
        "pde_map": pde_map.astype(np.float32),
        "dark_map": dark_map.astype(np.float32),
        "bg_spatial_map": bg_spatial.astype(np.float32),
        "total_signal_expected": int(round(total_signal)),
        "total_noise_expected": int(round(total_noise)),
        "observed_total_counts": int(np.sum(counts)),
        "snr_db": float(snr_db),
        "target_detected_rate_cps": float(np.mean(final_rate_cps_t)),
        "target_laser_detected_rate_cps": float(np.mean(final_laser_rate_cps_t)),
        "target_solar_detected_rate_cps": float(np.mean(final_solar_rate_cps_t)),
        "mean_signal_per_frame": mean_signal_per_frame,
        "mean_background_per_frame": mean_background_per_frame,
        "mean_dark_per_frame": mean_dark_per_frame,
        "total_signal_photons": total_signal,
        "total_background_photons": total_background,
        "total_noise_photons": total_noise,
        "pixel_ifov_urad": pixel_ifov_urad,
        "mean_projected_width_px": float(np.mean(projected_width_px_t)),
        "mean_projected_height_px": float(np.mean(projected_height_px_t)),
        "mean_projected_width_urad": float(np.mean(projected_width_urad_t)),
        "mean_projected_height_urad": float(np.mean(projected_height_urad_t)),
        "fov_clipping_ratio": float(1.0 - fov_visibility),
        "atmospheric_transmission_mean": atmospheric_transmission_mean,
        "atmospheric_path_length_m_mean": float(np.mean(atmospheric_path_length_m)),
        "mean_in_fov_ratio": float(np.mean(truth_in_fov_t)),
        "dead_time_loss_ratio": dead_time_loss_ratio,
        "saturation_warning": bool(np.any(saturation_mask) or np.max(ideal_rate) > params.spad.max_count_rate_cps_per_pixel),
        "visibility_ratio": float(np.mean(visibility_t > 0)),
        "dropout_ratio": float(1.0 - np.mean(visibility_t > 0)),
        "outage_mode": getattr(params.target, "outage_mode", "random_segments"),
        "harmonic_truth_strength": float(harmonic_truth_strength),
        "detector_summary": detector_summary,
        "params": asdict(params),
        "background_components": {
            "scene_stray": bg_scene_stray_t.astype(np.float32),
        },
    }
    return result
