"""Converters between API models, simulator config, and API responses."""

from __future__ import annotations

import numpy as np

from backend.capabilities import detect_compute_capabilities
from backend.models import SimulateRequest, SimulateResponse, SimulateSummaryResponse
from backend.serializers import encode_array
from sim.config import SimParams
from sim.detector_presets import apply_detector_preset, refresh_pf32_spectral_defaults


def _apply_if_set(req: SimulateRequest, field_name: str, setter) -> None:
    if field_name in req.model_fields_set:
        setter(getattr(req, field_name))


def _has_all_fields(req: SimulateRequest, field_names: tuple[str, ...]) -> bool:
    return all(name in req.model_fields_set and getattr(req, name) is not None for name in field_names)


def _apply_scene_geometry(req: SimulateRequest, params: SimParams) -> None:
    target_fields = ("target_position_x_m", "target_position_y_m", "target_position_z_m")
    if not _has_all_fields(req, target_fields):
        return

    detector_x = float(req.detector_position_x_m) if req.detector_position_x_m is not None else 0.0
    detector_y = float(req.detector_position_y_m) if req.detector_position_y_m is not None else 0.0
    detector_z = float(req.detector_position_z_m) if req.detector_position_z_m is not None else 0.0
    rel_x = float(req.target_position_x_m) - detector_x
    rel_y = float(req.target_position_y_m) - detector_y
    rel_z = float(req.target_position_z_m) - detector_z
    range_m = float(np.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z))
    if range_m <= 0:
        return

    bearing_yaw_deg = float(np.rad2deg(np.arctan2(rel_x, rel_z)))
    ground_range = float(np.hypot(rel_x, rel_z))
    bearing_pitch_deg = float(np.rad2deg(np.arctan2(rel_y, max(ground_range, 1e-9))))
    detector_yaw_deg = float(req.detector_yaw_deg) if req.detector_yaw_deg is not None else 0.0
    detector_pitch_deg = float(req.detector_pitch_deg) if req.detector_pitch_deg is not None else 0.0
    params.geometry.detector_position_x_m = detector_x
    params.geometry.detector_position_y_m = detector_y
    params.geometry.detector_position_z_m = detector_z
    params.geometry.detector_yaw_deg = detector_yaw_deg
    params.geometry.detector_pitch_deg = detector_pitch_deg
    params.geometry.external_target_x_m = np.asarray([float(req.target_position_x_m)], dtype=np.float64)
    params.geometry.external_target_y_m = np.asarray([float(req.target_position_y_m)], dtype=np.float64)
    params.geometry.external_target_z_m = np.asarray([float(req.target_position_z_m)], dtype=np.float64)

    params.target.target_range_m = max(range_m, 1.0)
    params.target.reference_range_m = max(range_m, 1.0)
    params.optical.detector_off_axis_urad_x = float(np.deg2rad(bearing_yaw_deg - detector_yaw_deg) * 1e6)
    params.optical.detector_off_axis_urad_y = float(-np.deg2rad(bearing_pitch_deg - detector_pitch_deg) * 1e6)
    params.geometry.los_azimuth_deg = bearing_yaw_deg
    params.geometry.los_elevation_deg = bearing_pitch_deg
    pixel_axis = float(max(params.image.roi_w, params.image.roi_h, 1))
    pixel_ifov_urad = max(float(params.optical.detector_fov_urad) / pixel_axis, 1e-6)
    params.image.scene_mode = "moving_target_in_fov"
    params.image.initial_center_x = params.image.center_x + params.optical.detector_off_axis_urad_x / pixel_ifov_urad
    params.image.initial_center_y = params.image.center_y + params.optical.detector_off_axis_urad_y / pixel_ifov_urad

    if req.target_yaw_deg is not None:
        params.target.orientation_yaw_deg = float(req.target_yaw_deg)
        params.target.spin_axis_azimuth_deg = float(req.target_yaw_deg)
        params.geometry.external_yaw_deg = np.asarray([float(req.target_yaw_deg)], dtype=np.float64)
    if req.target_pitch_deg is not None:
        params.target.orientation_pitch_deg = float(req.target_pitch_deg)
        params.target.spin_axis_elevation_deg = float(np.clip(req.target_pitch_deg + 45.0, -89.0, 89.0))
        params.geometry.external_pitch_deg = np.asarray([float(req.target_pitch_deg)], dtype=np.float64)
    if req.target_roll_deg is not None:
        params.target.orientation_roll_deg = float(req.target_roll_deg)
        params.target.phase1 = float(np.deg2rad(req.target_roll_deg))
        params.geometry.external_roll_deg = np.asarray([float(req.target_roll_deg)], dtype=np.float64)


def _apply_recorded_trajectory(req: SimulateRequest, params: SimParams) -> None:
    trajectory_fields = (
        "target_trajectory_times_s",
        "target_trajectory_x_m",
        "target_trajectory_y_m",
        "target_trajectory_z_m",
    )
    if not _has_all_fields(req, trajectory_fields):
        return

    source_t = np.asarray(req.target_trajectory_times_s, dtype=np.float64)
    source_x = np.asarray(req.target_trajectory_x_m, dtype=np.float64)
    source_y = np.asarray(req.target_trajectory_y_m, dtype=np.float64)
    source_z = np.asarray(req.target_trajectory_z_m, dtype=np.float64)
    n_source = int(source_t.size)
    if n_source < 2 or not (source_x.size == source_y.size == source_z.size == n_source):
        return

    order = np.argsort(source_t)
    source_t = source_t[order]
    source_x = source_x[order]
    source_y = source_y[order]
    source_z = source_z[order]
    source_t = source_t - float(source_t[0])
    unique_mask = np.r_[True, np.diff(source_t) > 1e-6]
    source_t = source_t[unique_mask]
    source_x = source_x[unique_mask]
    source_y = source_y[unique_mask]
    source_z = source_z[unique_mask]
    if source_t.size < 2:
        return

    duration_s = max(float(source_t[-1]), 1e-6)
    dt = 1.0 / max(float(params.sample_rate_hz), 1e-9)
    params.observation_time_s = max(float(params.observation_time_s), duration_s + dt)
    n_frames = max(1, int(round(params.observation_time_s * params.sample_rate_hz)))
    t = np.arange(n_frames, dtype=np.float64) / max(float(params.sample_rate_hz), 1e-9)
    t = np.clip(t, 0.0, duration_s)

    detector_x = float(req.detector_position_x_m) if req.detector_position_x_m is not None else 0.0
    detector_y = float(req.detector_position_y_m) if req.detector_position_y_m is not None else 0.0
    detector_z = float(req.detector_position_z_m) if req.detector_position_z_m is not None else 0.0
    params.geometry.detector_position_x_m = detector_x
    params.geometry.detector_position_y_m = detector_y
    params.geometry.detector_position_z_m = detector_z
    x_t = np.interp(t, source_t, source_x)
    y_t = np.interp(t, source_t, source_y)
    z_t = np.interp(t, source_t, source_z)
    rel = np.stack([x_t - detector_x, y_t - detector_y, z_t - detector_z], axis=-1)
    range_t = np.maximum(np.linalg.norm(rel, axis=-1), 1.0)
    rel_unit = rel / range_t[:, None]

    detector_yaw_deg = float(req.detector_yaw_deg) if req.detector_yaw_deg is not None else 0.0
    detector_pitch_deg = float(req.detector_pitch_deg) if req.detector_pitch_deg is not None else 0.0
    params.geometry.detector_yaw_deg = detector_yaw_deg
    params.geometry.detector_pitch_deg = detector_pitch_deg
    bearing_yaw_deg = np.rad2deg(np.arctan2(rel[:, 0], rel[:, 2]))
    ground_range = np.hypot(rel[:, 0], rel[:, 2])
    bearing_pitch_deg = np.rad2deg(np.arctan2(rel[:, 1], np.maximum(ground_range, 1e-9)))
    off_axis_x = np.deg2rad(bearing_yaw_deg - detector_yaw_deg) * 1e6
    off_axis_y = -np.deg2rad(bearing_pitch_deg - detector_pitch_deg) * 1e6
    half_fov = max(float(params.optical.detector_fov_urad) / 2.0, 1e-6)
    in_fov = np.hypot(off_axis_x, off_axis_y) <= half_fov

    params.target.target_range_m = float(range_t[0])
    params.target.reference_range_m = float(range_t[0])
    params.optical.detector_off_axis_urad_x = float(off_axis_x[0])
    params.optical.detector_off_axis_urad_y = float(off_axis_y[0])
    params.image.scene_mode = "moving_target_in_fov"
    params.geometry.trajectory_name = "manual_recorded_flight"
    params.geometry.external_range_m = range_t.astype(np.float64)
    params.geometry.external_target_x_m = x_t.astype(np.float64)
    params.geometry.external_target_y_m = y_t.astype(np.float64)
    params.geometry.external_target_z_m = z_t.astype(np.float64)
    params.geometry.external_los_unit = rel_unit.astype(np.float64)
    params.geometry.external_sun_unit = np.broadcast_to(
        np.array([1.0, 0.0, 0.35], dtype=np.float64) / np.linalg.norm([1.0, 0.0, 0.35]),
        rel_unit.shape,
    ).copy()
    params.geometry.external_phase_angle_rad = None
    params.geometry.external_off_axis_x_urad = off_axis_x.astype(np.float64)
    params.geometry.external_off_axis_y_urad = off_axis_y.astype(np.float64)
    params.geometry.external_in_fov = in_fov.astype(np.float64)
    params.geometry.external_visibility = np.ones(n_frames, dtype=np.float64)

    if req.target_trajectory_yaw_deg and len(req.target_trajectory_yaw_deg) == n_source:
        yaw_t = np.asarray(req.target_trajectory_yaw_deg, dtype=np.float64)[order][unique_mask]
        yaw_frame_t = np.interp(t, source_t, yaw_t)
        params.geometry.external_yaw_deg = yaw_frame_t.astype(np.float64)
        params.target.orientation_yaw_deg = float(np.interp(0.0, source_t, yaw_t))
        params.target.spin_axis_azimuth_deg = float(np.interp(0.0, source_t, yaw_t))
    if req.target_trajectory_pitch_deg and len(req.target_trajectory_pitch_deg) == n_source:
        pitch_t = np.asarray(req.target_trajectory_pitch_deg, dtype=np.float64)[order][unique_mask]
        pitch_frame_t = np.interp(t, source_t, pitch_t)
        params.geometry.external_pitch_deg = pitch_frame_t.astype(np.float64)
        params.target.orientation_pitch_deg = float(np.interp(0.0, source_t, pitch_t))
        params.target.spin_axis_elevation_deg = float(np.clip(np.interp(0.0, source_t, pitch_t) + 45.0, -89.0, 89.0))
    if req.target_trajectory_roll_deg and len(req.target_trajectory_roll_deg) == n_source:
        roll_t = np.asarray(req.target_trajectory_roll_deg, dtype=np.float64)[order][unique_mask]
        roll_frame_t = np.interp(t, source_t, roll_t)
        params.geometry.external_roll_deg = roll_frame_t.astype(np.float64)
        params.target.orientation_roll_deg = float(np.interp(0.0, source_t, roll_t))
        params.target.phase1 = float(np.deg2rad(np.interp(0.0, source_t, roll_t)))
    if req.target_trajectory_phase_rad and len(req.target_trajectory_phase_rad) == n_source:
        phase_t = np.asarray(req.target_trajectory_phase_rad, dtype=np.float64)[order][unique_mask]
        params.geometry.external_rotation_phase_rad = np.interp(t, source_t, phase_t).astype(np.float64)
    rpm_series = []
    for field_name in (
        "target_trajectory_propeller_rpm1",
        "target_trajectory_propeller_rpm2",
        "target_trajectory_propeller_rpm3",
        "target_trajectory_propeller_rpm4",
    ):
        values = getattr(req, field_name)
        if values and len(values) == n_source:
            rpm_series.append(np.asarray(values, dtype=np.float64)[order][unique_mask])
    if rpm_series:
        rpm_frame_t = np.mean(np.stack([np.interp(t, source_t, rpm) for rpm in rpm_series], axis=0), axis=0)
        params.geometry.external_spin_hz = (rpm_frame_t / 60.0).astype(np.float64)
        mean_rpm = float(np.mean(rpm_frame_t))
        if np.isfinite(mean_rpm) and mean_rpm > 0:
            params.target.spin_hz = mean_rpm / 60.0
        params.geometry.external_propeller_spin_hz = np.stack(
            [np.interp(t, source_t, rpm) / 60.0 for rpm in rpm_series],
            axis=1,
        ).astype(np.float64)
    phase_series = []
    for field_name in (
        "target_trajectory_propeller_phase1_rad",
        "target_trajectory_propeller_phase2_rad",
        "target_trajectory_propeller_phase3_rad",
        "target_trajectory_propeller_phase4_rad",
    ):
        values = getattr(req, field_name)
        if values and len(values) == n_source:
            phase_series.append(np.asarray(values, dtype=np.float64)[order][unique_mask])
    if phase_series:
        params.geometry.external_propeller_phase_rad = np.stack(
            [np.interp(t, source_t, phase) for phase in phase_series],
            axis=1,
        ).astype(np.float64)


def params_from_request(req: SimulateRequest) -> tuple[SimParams, dict | None]:
    """Convert a frontend request into simulator parameters."""

    scenario_info = None
    params = SimParams()
    apply_detector_preset(params, req.detector_preset)

    _apply_if_set(req, "detector_preset", lambda v: apply_detector_preset(params, v))
    _apply_if_set(req, "observation_time_s", lambda v: setattr(params, "observation_time_s", v))
    _apply_if_set(req, "sample_rate_hz", lambda v: setattr(params, "sample_rate_hz", v))
    _apply_if_set(req, "seed", lambda v: setattr(params, "seed", v))
    _apply_if_set(req, "compute_backend", lambda v: setattr(params, "compute_backend", v))
    _apply_if_set(req, "simulation_tier", lambda v: setattr(params, "simulation_tier", v))
    _apply_if_set(req, "output_mode", lambda v: setattr(params, "simulation_mode", v))
    _apply_if_set(req, "lightcurve_mode", lambda v: setattr(params, "lightcurve_mode", v))
    _apply_if_set(req, "save_truth_series", lambda v: setattr(params, "save_truth_series", v))

    _apply_if_set(req, "target_range_m", lambda v: setattr(params.target, "target_range_m", v))
    _apply_if_set(req, "target_radial_velocity_mps", lambda v: setattr(params.target, "target_radial_velocity_mps", v))
    _apply_if_set(req, "target_radial_accel_mps2", lambda v: setattr(params.target, "target_radial_accel_mps2", v))
    _apply_if_set(req, "target_area_m2", lambda v: setattr(params.target, "target_area_m2", v))
    _apply_if_set(req, "target_length_m", lambda v: setattr(params.target, "target_length_m", v))
    _apply_if_set(req, "target_width_m", lambda v: setattr(params.target, "target_width_m", v))
    _apply_if_set(req, "target_height_m", lambda v: setattr(params.target, "target_height_m", v))
    _apply_if_set(req, "propeller_diameter_m", lambda v: setattr(params.target, "propeller_diameter_m", v))
    _apply_if_set(req, "target_reflectivity", lambda v: setattr(params.target, "target_reflectivity", v))
    _apply_if_set(req, "propeller_reflectivity", lambda v: setattr(params.target, "propeller_reflectivity", v))
    _apply_if_set(req, "solar_irradiance", lambda v: setattr(params.target, "solar_irradiance_w_m2_nm", v))
    _apply_if_set(req, "phase_function_scale", lambda v: setattr(params.target, "phase_function_scale", v))
    _apply_if_set(req, "specular_fraction", lambda v: setattr(params.target, "specular_fraction", v))
    _apply_if_set(req, "modulation_depth", lambda v: setattr(params.target, "modulation_depth", v))
    _apply_if_set(req, "tumbling_hz", lambda v: setattr(params.target, "tumbling_hz", v))
    _apply_if_set(req, "spin_hz", lambda v: setattr(params.target, "spin_hz", v))
    _apply_if_set(req, "precession_hz", lambda v: setattr(params.target, "precession_hz", v))
    _apply_if_set(req, "body_shape", lambda v: setattr(params.target, "body_shape", v))
    _apply_if_set(req, "custom_shape_x", lambda v: setattr(params.target, "custom_shape_x", np.asarray(v, dtype=np.float64)))
    _apply_if_set(req, "custom_shape_y", lambda v: setattr(params.target, "custom_shape_y", np.asarray(v, dtype=np.float64)))
    _apply_if_set(req, "custom_shape_intensity", lambda v: setattr(params.target, "custom_shape_intensity", np.asarray(v, dtype=np.float64)))
    _apply_if_set(req, "custom_shape_aspect_ratio", lambda v: setattr(params.target, "custom_shape_aspect_ratio", float(v)))
    _apply_if_set(req, "outage_fraction", lambda v: setattr(params.target, "outage_fraction", v))
    _apply_if_set(req, "glint_probability", lambda v: setattr(params.target, "glint_probability", v))
    _apply_if_set(req, "glint_gain", lambda v: setattr(params.target, "glint_gain", v))

    _apply_if_set(req, "scene_stray_rate", lambda v: setattr(params.background, "scene_stray_rate_cps_per_pixel", v))

    _apply_if_set(req, "illumination_mode", lambda v: setattr(params.illumination, "mode", v))
    _apply_if_set(req, "laser_mode", lambda v: setattr(params.illumination, "laser_mode", v))
    _apply_if_set(req, "laser_average_power_w", lambda v: setattr(params.illumination, "laser_average_power_w", v))
    _apply_if_set(req, "laser_pulse_energy_j", lambda v: setattr(params.illumination, "laser_pulse_energy_j", v))
    _apply_if_set(req, "laser_repetition_frequency_hz", lambda v: setattr(params.illumination, "laser_repetition_frequency_hz", v))
    _apply_if_set(req, "laser_pulse_width_ns", lambda v: setattr(params.illumination, "laser_pulse_width_ns", v))
    _apply_if_set(req, "transmitter_divergence_mrad", lambda v: setattr(params.illumination, "transmitter_divergence_mrad", v))

    _apply_if_set(req, "aperture_diameter_m", lambda v: setattr(params.optical, "aperture_diameter_m", v))
    _apply_if_set(req, "receiver_efficiency", lambda v: setattr(params.optical, "receiver_efficiency", v))
    _apply_if_set(req, "quantum_efficiency", lambda v: setattr(params.optical, "quantum_efficiency", v))
    _apply_if_set(req, "wavelength_nm", lambda v: setattr(params.optical, "wavelength_nm", v))
    _apply_if_set(req, "filter_bandwidth_nm", lambda v: setattr(params.optical, "filter_bandwidth_nm", v))
    _apply_if_set(req, "detector_fov_urad", lambda v: setattr(params.optical, "detector_fov_urad", v))
    _apply_if_set(req, "atmospheric_attenuation_enabled", lambda v: setattr(params.optical, "atmospheric_attenuation_enabled", v))
    _apply_if_set(req, "atmospheric_visibility_km", lambda v: setattr(params.optical, "atmospheric_visibility_km", v))

    _apply_if_set(req, "dark_count_rate", lambda v: setattr(params.spad, "dark_count_rate_cps", v))
    _apply_if_set(req, "dead_time_ns", lambda v: setattr(params.spad, "dead_time_ns", v))
    _apply_if_set(req, "max_count_per_frame", lambda v: setattr(params.spad, "max_count_per_frame", v))
    _apply_if_set(req, "timing_jitter_ns", lambda v: setattr(params.spad, "timing_jitter_ns", v))
    _apply_if_set(req, "tdc_bin_width_ns", lambda v: setattr(params.spad, "tdc_bin_width_ns", v))
    _apply_if_set(req, "irf_fwhm_ps", lambda v: setattr(params.spad, "irf_fwhm_ps", v))
    _apply_if_set(req, "max_count_rate_cps_per_pixel", lambda v: setattr(params.spad, "max_count_rate_cps_per_pixel", v))
    _apply_if_set(req, "detector_off_axis_urad_x", lambda v: setattr(params.optical, "detector_off_axis_urad_x", v))
    _apply_if_set(req, "detector_off_axis_urad_y", lambda v: setattr(params.optical, "detector_off_axis_urad_y", v))

    _apply_if_set(req, "roi_w", lambda v: setattr(params.image, "roi_w", v))
    _apply_if_set(req, "roi_h", lambda v: setattr(params.image, "roi_h", v))
    _apply_if_set(req, "pixel_pitch_um", lambda v: setattr(params.image, "pixel_pitch_um", v))
    _apply_if_set(req, "fill_factor", lambda v: setattr(params.image, "fill_factor", v))
    _apply_if_set(req, "microlens_gain", lambda v: setattr(params.image, "microlens_gain", v))
    _apply_if_set(req, "drift_pixels_per_s_x", lambda v: setattr(params.image, "drift_pixels_per_s_x", v))
    _apply_if_set(req, "drift_pixels_per_s_y", lambda v: setattr(params.image, "drift_pixels_per_s_y", v))
    _apply_if_set(req, "jitter_sigma_pixels", lambda v: setattr(params.image, "jitter_sigma_pixels", v))
    _apply_scene_geometry(req, params)
    _apply_recorded_trajectory(req, params)

    if params.image.center_x < 0 or params.image.center_x >= params.image.roi_w:
        params.image.center_x = (params.image.roi_w - 1) / 2.0
    if params.image.center_y < 0 or params.image.center_y >= params.image.roi_h:
        params.image.center_y = (params.image.roi_h - 1) / 2.0

    refresh_pf32_spectral_defaults(
        params,
        update_quantum_efficiency="quantum_efficiency" not in req.model_fields_set,
        update_solar_irradiance=(
            "wavelength_nm" in req.model_fields_set
            and "solar_irradiance" not in req.model_fields_set
        ),
    )

    return params, scenario_info


def result_to_response(
    result: dict,
    scenario_info: dict | None,
) -> SimulateResponse:
    """Convert simulator result dict into API response."""

    event_times = result.get("event_times")
    event_pixels = result.get("event_pixels")
    return SimulateResponse(
        scenario_id=result.get("scenario_id"),
        scenario_name=scenario_info["name"] if scenario_info else None,
        simulation_tier=result["simulation_tier"],
        output_mode=result["output_mode"],
        lightcurve_mode=result["lightcurve_mode"],
        assumptions=result.get("assumptions", []),
        warnings=result.get("warnings", []),
        detector={
            "preset": result["detector_summary"]["preset"],
            "preset_label": result["detector_summary"]["preset_label"],
            "array_rows": result["detector_summary"]["array_rows"],
            "array_cols": result["detector_summary"]["array_cols"],
            "pixel_pitch_um": result["detector_summary"]["pixel_pitch_um"],
            "fill_factor": result["detector_summary"]["fill_factor"],
            "microlens_gain": result["detector_summary"]["microlens_gain"],
            "detector_fov_urad": result["detector_summary"]["detector_fov_urad"],
            "pixel_ifov_urad": result["detector_summary"]["pixel_ifov_urad"],
            "tdc_bin_width_ns": result["detector_summary"]["tdc_bin_width_ns"],
            "irf_fwhm_ps": result["detector_summary"]["irf_fwhm_ps"],
            "max_count_rate_cps_per_pixel": result["detector_summary"]["max_count_rate_cps_per_pixel"],
            "max_count_per_frame": result["detector_summary"]["max_count_per_frame"],
            "quantum_efficiency": result["detector_summary"]["quantum_efficiency"],
            "receiver_efficiency": result["detector_summary"]["receiver_efficiency"],
            "assumptions": result["detector_summary"]["assumptions"],
        },
        snr_db=result["snr_db"],
        truth_freq_hz=result["truth_freq_hz"],
        truth_pixel=result["truth_pixel"],
        truth_row=result["truth_row"],
        truth_col=result["truth_col"],
        n_frames=result["n_frames"],
        roi_h=result["roi_h"],
        roi_w=result["roi_w"],
        sample_rate_hz=result["sample_rate_hz"],
        target_detected_rate_cps=result["target_detected_rate_cps"],
        target_laser_detected_rate_cps=result["target_laser_detected_rate_cps"],
        target_solar_detected_rate_cps=result["target_solar_detected_rate_cps"],
        mean_signal_per_frame=result["mean_signal_per_frame"],
        mean_background_per_frame=result["mean_background_per_frame"],
        mean_dark_per_frame=result["mean_dark_per_frame"],
        total_signal_photons=result["total_signal_photons"],
        total_background_photons=result["total_background_photons"],
        total_noise_photons=result["total_noise_photons"],
        pixel_ifov_urad=result["pixel_ifov_urad"],
        fov_clipping_ratio=result["fov_clipping_ratio"],
        dead_time_loss_ratio=result["dead_time_loss_ratio"],
        saturation_warning=result["saturation_warning"],
        visibility_ratio=result["visibility_ratio"],
        dropout_ratio=result["dropout_ratio"],
        harmonic_truth_strength=result["harmonic_truth_strength"],
        event_count=int(len(event_times)) if event_times is not None else 0,
        counts_encoded=encode_array(result["counts"]),
        event_times_encoded=encode_array(np.asarray(event_times, dtype=np.float32)) if event_times is not None else None,
        event_pixels_encoded=encode_array(np.asarray(event_pixels, dtype=np.int32)) if event_pixels is not None else None,
        expected_signal_map_encoded=encode_array(result["expected_signal_map"]) if result.get("expected_signal_map") is not None else None,
        truth_signal_series_encoded=encode_array(result["truth_signal_series"]) if result["truth_signal_series"] is not None else None,
        truth_bg_base_series_encoded=encode_array(result["truth_bg_base_series"]) if result["truth_bg_base_series"] is not None else None,
        truth_bg_total_series_encoded=encode_array(result["truth_bg_total_series"]) if result["truth_bg_total_series"] is not None else None,
        truth_visibility_series_encoded=encode_array(result["truth_visibility_series"]) if result["truth_visibility_series"] is not None else None,
        truth_cx_series_encoded=encode_array(result["truth_cx_series"]) if result["truth_cx_series"] is not None else None,
        truth_cy_series_encoded=encode_array(result["truth_cy_series"]) if result["truth_cy_series"] is not None else None,
        truth_projected_width_px_series_encoded=encode_array(result["truth_projected_width_px_series"]) if result["truth_projected_width_px_series"] is not None else None,
        truth_projected_height_px_series_encoded=encode_array(result["truth_projected_height_px_series"]) if result["truth_projected_height_px_series"] is not None else None,
        truth_range_series_encoded=encode_array(result["truth_range_series"]) if result["truth_range_series"] is not None else None,
        truth_glint_series_encoded=encode_array(result["truth_glint_series"]) if result["truth_glint_series"] is not None else None,
        modulation_series_encoded=encode_array(result["modulation_series"]) if result["modulation_series"] is not None else None,
        pde_map_encoded=encode_array(result["pde_map"]),
        dark_map_encoded=encode_array(result["dark_map"]),
        bg_spatial_map_encoded=encode_array(result["bg_spatial_map"]),
    )


def result_to_summary_response(
    result: dict,
    scenario_info: dict | None,
) -> SimulateSummaryResponse:
    """Convert simulator result into a lightweight frontend summary."""

    counts = np.asarray(result["counts"])
    preview = np.sum(counts, axis=0).astype(np.int64)
    expected_signal_map = np.asarray(result.get("expected_signal_map"), dtype=np.float64)
    if expected_signal_map.shape != preview.shape:
        expected_signal_map = np.zeros_like(preview, dtype=np.float64)
    sample_backend = result.get("sample_backend")
    capabilities = detect_compute_capabilities()
    actual_backend = sample_backend if sample_backend in {"cpu", "cuda"} else capabilities["recommended_backend"]
    return SimulateSummaryResponse(
        scenario_id=result.get("scenario_id"),
        scenario_name=scenario_info["name"] if scenario_info else None,
        simulation_tier=result["simulation_tier"],
        output_mode=result["output_mode"],
        lightcurve_mode=result["lightcurve_mode"],
        compute_backend=actual_backend,
        sample_backend=actual_backend,
        encoded_payload_omitted=True,
        n_frames=result["n_frames"],
        roi_h=result["roi_h"],
        roi_w=result["roi_w"],
        sample_rate_hz=result["sample_rate_hz"],
        snr_db=result["snr_db"],
        observed_total_counts=int(result.get("observed_total_counts", int(np.sum(counts)))),
        total_signal_photons=result["total_signal_photons"],
        total_background_photons=result["total_background_photons"],
        total_noise_photons=result["total_noise_photons"],
        mean_signal_per_frame=result["mean_signal_per_frame"],
        mean_background_per_frame=result["mean_background_per_frame"],
        mean_dark_per_frame=result["mean_dark_per_frame"],
        fov_clipping_ratio=result.get("fov_clipping_ratio", 0.0),
        mean_in_fov_ratio=result.get("mean_in_fov_ratio", 1.0),
        atmospheric_transmission_mean=result.get("atmospheric_transmission_mean", 1.0),
        dead_time_loss_ratio=result["dead_time_loss_ratio"],
        saturation_warning=result["saturation_warning"],
        visibility_ratio=result["visibility_ratio"],
        dropout_ratio=result["dropout_ratio"],
        target_detected_rate_cps=result["target_detected_rate_cps"],
        target_laser_detected_rate_cps=result["target_laser_detected_rate_cps"],
        target_solar_detected_rate_cps=result["target_solar_detected_rate_cps"],
        truth_freq_hz=result["truth_freq_hz"],
        truth_row=result["truth_row"],
        truth_col=result["truth_col"],
        preview_counts=preview.tolist(),
        expected_signal_map=expected_signal_map.tolist(),
        warnings=result.get("warnings", []),
        assumptions=result.get("assumptions", []),
    )
