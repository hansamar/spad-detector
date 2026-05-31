from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.convert import params_from_request, result_to_response
from backend.models import SimulateRequest
from sim.active_imaging_sim import (
    _event_stream_from_counts,
    _frontend_drone_world_points,
    _background_series,
    _project_world_point_to_pixel,
    _shot_noise_snr_db,
    _shape_signal_distribution_cube,
    simulate_active_spad,
)
from sim.background import background_spatial_map
from sim import detector as detector_module
from sim.config import SimParams
from sim.physics import laser_target_detected_rate_cps


FOV_URAD_50_DEG = 50.0 * np.pi / 180.0 * 1e6


def _base_request(**overrides):
    data = {
        "detector_preset": "pf32",
        "observation_time_s": 0.04,
        "sample_rate_hz": 50,
        "seed": 7,
        "target_position_x_m": 0.0,
        "target_position_y_m": 1.0,
        "target_position_z_m": 1.5,
        "detector_position_x_m": 0.0,
        "detector_position_y_m": 1.0,
        "detector_position_z_m": 0.0,
        "detector_yaw_deg": 0.0,
        "detector_pitch_deg": 0.0,
        "detector_fov_urad": FOV_URAD_50_DEG,
        "roi_w": 32,
        "roi_h": 32,
        "target_reflectivity": 0.3,
        "dark_count_rate": 0,
        "scene_stray_rate": 0,
        "save_truth_series": True,
    }
    data.update(overrides)
    return SimulateRequest(**data)


def _params(**overrides):
    params, _scenario = params_from_request(_base_request(**overrides))
    return params


def _shape_cube(params, *, photons=1000.0):
    return _shape_signal_distribution_cube(
        np.ones(2, dtype=np.float64) * photons,
        np.ones(2, dtype=np.float64) * 16.0,
        np.ones(2, dtype=np.float64) * 16.0,
        32,
        32,
        np.ones(2, dtype=np.float64) * 4.0,
        np.ones(2, dtype=np.float64) * 4.0,
        np.array([0.0, 0.1], dtype=np.float64),
        params,
        0.1,
        0.1,
        np.ones((32, 32), dtype=np.float64),
    )


def test_frontend_projection_y_axis() -> None:
    params = _params(body_shape="sphere", target_width_m=0.067)
    low = _project_world_point_to_pixel((0.0, 1.0, 1.5), params, 32, 32)
    high = _project_world_point_to_pixel((0.0, 2.0, 1.5), params, 32, 32)
    assert low is not None and high is not None
    assert high[1] < low[1], f"Y projection flipped: low={low[1]:.3f}, high={high[1]:.3f}"


def test_scene_stray_background_has_stationary_floor_without_space_terms() -> None:
    params = _params(
        scene_stray_rate=20,
    )
    t = np.linspace(0, 0.04, 4)
    phase = np.zeros_like(t)
    los = np.tile(np.array([[0.0, 0.0, 1.0]]), (4, 1))
    total, scene_stray = _background_series(t, phase, los, params)
    assert np.all(total > 0)
    assert float(np.mean(total)) >= 7.0
    assert np.allclose(total, scene_stray)


def test_background_spatial_map_stays_positive_after_large_gradients() -> None:
    spatial = background_spatial_map(4, 4, gradient_x=1.0, gradient_y=1.0)
    assert np.all(spatial > 0), f"background map contains non-positive weights: min={float(np.min(spatial)):.3g}"
    assert np.isclose(float(np.mean(spatial)), 1.0)


def test_recorded_trajectory_uses_circular_detector_fov() -> None:
    angle_rad = np.deg2rad(20.0)
    z_m = 1.0
    x_m = float(np.tan(angle_rad) * z_m)
    ground_range_m = float(np.hypot(x_m, z_m))
    y_m = float(np.tan(angle_rad) * ground_range_m)
    params, _scenario = params_from_request(
        _base_request(
            observation_time_s=0.02,
            target_trajectory_times_s=[0.0, 0.02],
            target_trajectory_x_m=[x_m, x_m],
            target_trajectory_y_m=[y_m, y_m],
            target_trajectory_z_m=[z_m, z_m],
        )
    )
    assert params.geometry.external_in_fov is not None
    assert not np.any(params.geometry.external_in_fov), (
        "trajectory point outside the circular FOV was accepted by independent x/y bounds"
    )


def test_event_output_matches_sampled_frame_counts() -> None:
    params, _scenario = params_from_request(
        _base_request(
            output_mode="event",
            observation_time_s=0.2,
            sample_rate_hz=50,
            scene_stray_rate=100,
            dark_count_rate=20,
            solar_irradiance=0,
            target_reflectivity=0,
        )
    )
    result = simulate_active_spad(params)
    assert result["event_times"] is not None
    assert int(result["event_times"].size) == int(np.sum(result["counts"])), (
        "event output independently resampled the already emitted frame counts"
    )


def test_event_timestamps_remain_inside_source_frame_after_jitter() -> None:
    class FixedRng:
        @staticmethod
        def uniform(low: float, high: float, size: int) -> np.ndarray:
            return np.full(size, high * 0.9, dtype=np.float64)

        @staticmethod
        def normal(mean: float, sigma: float, size: int) -> np.ndarray:
            return np.full(size, sigma * 10.0, dtype=np.float64)

    event_times, _event_pixels = _event_stream_from_counts(
        np.ones((1, 1, 1), dtype=np.uint16),
        dt=1e-9,
        timing_jitter_ns=100.0,
        tdc_bin_width_ns=0.0,
        rng=FixedRng(),
    )
    assert event_times.size == 1
    assert 0 <= float(event_times[0]) < 1e-9


def test_pf32_timing_defaults_match_public_irf_specification() -> None:
    params = _params()
    assert np.isclose(params.spad.irf_fwhm_ps, 200.0)
    assert np.isclose(params.spad.timing_jitter_ns, 0.2 / 2.355)


def test_direct_python_defaults_match_pf32_detector() -> None:
    params = SimParams()
    assert params.detector_preset == "pf32"
    assert np.isclose(params.spad.timing_jitter_ns, 0.2 / 2.355)
    assert np.isclose(params.spad.tdc_bin_width_ns, 0.055)
    assert np.isclose(params.spad.irf_fwhm_ps, 200.0)
    assert np.isclose(params.spad.max_count_rate_cps_per_pixel, 20e6)


def test_illumination_request_keeps_transmitter_sun_and_scene_stray_independent() -> None:
    params = _params(
        illumination_mode="laser_plus_solar",
        laser_mode="pulsed",
        laser_average_power_w=0.25,
        laser_pulse_energy_j=2.5e-7,
        laser_repetition_frequency_hz=1e6,
        laser_pulse_width_ns=3.0,
        transmitter_divergence_mrad=0.7,
        solar_irradiance=0.42,
        scene_stray_rate=123.0,
    )
    assert params.illumination.mode == "laser_plus_solar"
    assert params.illumination.laser_mode == "pulsed"
    assert np.isclose(params.illumination.laser_average_power_w, 0.25)
    assert np.isclose(params.illumination.laser_pulse_energy_j, 2.5e-7)
    assert np.isclose(params.illumination.laser_repetition_frequency_hz, 1e6)
    assert np.isclose(params.illumination.laser_pulse_width_ns, 3.0)
    assert np.isclose(params.illumination.transmitter_divergence_mrad, 0.7)
    assert np.isclose(params.target.solar_irradiance_w_m2_nm, 0.42)
    assert np.isclose(params.background.scene_stray_rate_cps_per_pixel, 123.0)


def test_default_transmitter_starts_in_near_range_single_photon_regime() -> None:
    params = SimParams()
    assert params.illumination.mode == "laser_plus_solar"
    assert np.isclose(params.illumination.laser_average_power_w, 1e-6)
    assert np.isclose(params.illumination.laser_pulse_energy_j, 1e-12)
    assert np.isclose(params.illumination.transmitter_divergence_mrad, 1.0)


def test_laser_signal_rate_uses_transmitter_divergence_and_caps_intercepted_power() -> None:
    common = {
        "laser_mode": "cw",
        "laser_average_power_w": 0.1,
        "laser_pulse_energy_j": 0.0,
        "laser_repetition_frequency_hz": 1e6,
        "target_area_m2": 0.025,
        "target_reflectivity": 0.3,
        "phase_function_scale": 1.0,
        "range_m": 10.0,
        "aperture_diameter_m": 0.025,
        "wavelength_nm": 780.0,
        "receiver_efficiency": 0.05,
        "quantum_efficiency": 0.1,
    }
    narrow = laser_target_detected_rate_cps(transmitter_divergence_mrad=0.1, **common)
    wide = laser_target_detected_rate_cps(transmitter_divergence_mrad=100.0, **common)
    assert narrow > wide > 0

    saturated_area = laser_target_detected_rate_cps(
        transmitter_divergence_mrad=0.1,
        **{**common, "target_area_m2": 1e6},
    )
    assert np.isclose(saturated_area, narrow), "large target exceeded the transmitted laser-power budget"


def test_hybrid_signal_budget_is_sum_of_laser_and_solar_components() -> None:
    common = {
        "observation_time_s": 0.02,
        "sample_rate_hz": 50,
        "scene_stray_rate": 0,
        "dark_count_rate": 0,
        "solar_irradiance": 0.001,
        "laser_mode": "cw",
        "laser_average_power_w": 1e-6,
        "transmitter_divergence_mrad": 10.0,
    }

    def run_mode(mode: str) -> dict:
        params = _params(illumination_mode=mode, **common)
        return simulate_active_spad(params)

    solar = run_mode("solar")
    laser = run_mode("laser")
    hybrid = run_mode("laser_plus_solar")
    assert hybrid["target_laser_detected_rate_cps"] > 0
    assert hybrid["target_solar_detected_rate_cps"] > 0
    assert np.isclose(
        hybrid["target_detected_rate_cps"],
        laser["target_detected_rate_cps"] + solar["target_detected_rate_cps"],
    )


def test_attitude_driven_laser_return_does_not_reuse_solar_direction() -> None:
    def run_phase(phase_deg: float) -> dict:
        params = _params(
            body_shape="plate",
            illumination_mode="laser",
            laser_mode="cw",
            laser_average_power_w=1e-6,
            transmitter_divergence_mrad=1.0,
        )
        params.geometry.phase_angle_override_deg = phase_deg
        return simulate_active_spad(params)

    phase_zero = run_phase(0.0)
    phase_eighty = run_phase(80.0)
    assert np.isclose(
        phase_zero["target_laser_detected_rate_cps"],
        phase_eighty["target_laser_detected_rate_cps"],
    ), "pure laser return changed when only the solar direction changed"


def test_reported_snr_uses_photon_shot_noise() -> None:
    assert np.isclose(_shot_noise_snr_db(100.0, 0.0), 20.0)
    assert np.isclose(_shot_noise_snr_db(100.0, 100.0), 20.0 * np.log10(100.0 / np.sqrt(200.0)))


def test_explicit_cuda_request_does_not_silently_fallback() -> None:
    original_cuda_available = detector_module._cuda_available
    try:
        detector_module._cuda_available = lambda: False
        try:
            detector_module.sample_poisson_counts_accelerated(
                np.ones((1, 1, 1), dtype=np.float64),
                np.random.default_rng(1),
                requested_backend="cuda",
                seed=1,
            )
        except RuntimeError as exc:
            assert "CUDA backend requested" in str(exc)
        else:
            raise AssertionError("explicit CUDA request silently fell back to CPU")
    finally:
        detector_module._cuda_available = original_cuda_available


def test_custom_blade_uses_shape_points_not_center_blob() -> None:
    params = _params(
        body_shape="blade_strip",
        target_length_m=0.45,
        target_width_m=0.05,
        spin_hz=0,
        target_pitch_deg=0,
        custom_shape_x=[-0.45, 0.45],
        custom_shape_y=[0.0, 0.0],
        custom_shape_intensity=[1.0, 1.0],
        custom_shape_aspect_ratio=4.0,
    )
    cube = _shape_cube(params)
    frame = cube[0]
    left_energy = float(np.sum(frame[:, :16]))
    right_energy = float(np.sum(frame[:, 16:]))
    center_energy = float(np.sum(frame[14:19, 14:19]))
    assert left_energy > 0 and right_energy > 0, "custom blade should form separated projected lobes"
    assert center_energy < 0.25 * float(np.sum(frame)), "custom blade collapsed to center after PSF blur"


def test_zero_rpm_keeps_blade_orientation_static() -> None:
    params = _params(
        body_shape="blade_strip",
        target_length_m=0.45,
        target_width_m=0.05,
        spin_hz=0,
        target_pitch_deg=20,
    )
    cube = _shape_cube(params)
    assert np.allclose(cube[0], cube[1]), "zero RPM blade changed projected orientation between frames"


def test_path_phase_drives_projected_blade_orientation() -> None:
    params = _params(
        body_shape="blade_strip",
        target_length_m=0.45,
        target_width_m=0.05,
        spin_hz=0,
        target_pitch_deg=15,
        target_trajectory_times_s=[0.0, 0.1],
        target_trajectory_x_m=[0.0, 0.0],
        target_trajectory_y_m=[1.0, 1.0],
        target_trajectory_z_m=[1.5, 1.5],
        target_trajectory_phase_rad=[0.0, np.pi / 4.0],
    )
    cube = _shape_cube(params)
    assert not np.allclose(cube[0], cube[1]), "trajectory phase series is not changing backend blade projection"


def test_drone_four_propeller_centers_project_inside_detector() -> None:
    params = _params(
        body_shape="drone_quad",
        target_length_m=0.298,
        target_width_m=0.373,
        target_height_m=0.08,
        propeller_diameter_m=0.12,
        spin_hz=0,
        target_pitch_deg=25,
        target_roll_deg=12,
        target_yaw_deg=0,
    )
    target_pos = (0.0, 1.0, 1.5)
    arm_half_w = 0.373 * 0.36
    arm_half_l = 0.298 * 0.36
    centers = [
        (arm_half_w, arm_half_l),
        (-arm_half_w, arm_half_l),
        (arm_half_w, -arm_half_l),
        (-arm_half_w, -arm_half_l),
    ]
    world = _frontend_drone_world_points(target_pos, centers, np.deg2rad(25), 0.0, np.deg2rad(12))
    pixels = [_project_world_point_to_pixel(point, params, 32, 32) for point in world]
    assert all(pixel is not None for pixel in pixels), f"drone propeller centers left detector: {pixels}"
    rounded = {(round(pixel[0]), round(pixel[1])) for pixel in pixels if pixel is not None}
    assert len(rounded) == 4, f"four propeller centers collapsed in projection: {rounded}"


def test_fast_drone_propellers_use_rotor_disk_projection() -> None:
    static_params = _params(
        body_shape="drone_quad",
        target_length_m=0.298,
        target_width_m=0.373,
        target_height_m=0.08,
        propeller_diameter_m=0.12,
        spin_hz=0,
        target_pitch_deg=25,
        target_roll_deg=12,
    )
    fast_params = _params(
        body_shape="drone_quad",
        target_length_m=0.298,
        target_width_m=0.373,
        target_height_m=0.08,
        propeller_diameter_m=0.12,
        spin_hz=100,
        target_pitch_deg=25,
        target_roll_deg=12,
    )
    static_nonzero = int(np.count_nonzero(_shape_cube(static_params)[0]))
    fast_frame = _shape_cube(fast_params)[0]
    fast_nonzero = int(np.count_nonzero(fast_frame))
    assert fast_nonzero > 0, "fast rotor disk produced an empty projection"
    assert not np.allclose(_shape_cube(static_params)[0], fast_frame), (
        f"fast rotor disk should differ from static blade projection: static={static_nonzero}, fast={fast_nonzero}"
    )


def test_response_carries_backend_expected_signal_map() -> None:
    req = _base_request(
        body_shape="blade_strip",
        target_length_m=0.45,
        target_width_m=0.05,
        spin_hz=0,
        target_pitch_deg=20,
        observation_time_s=0.02,
    )
    params, scenario = params_from_request(req)
    result = simulate_active_spad(params)
    response = result_to_response(result, scenario)
    assert response.expected_signal_map_encoded is not None, "backend response missing expected signal map"
    assert result["expected_signal_map"].shape == (32, 32)
    assert float(np.sum(result["expected_signal_map"])) > 0


def test_default_python_scene_uses_near_range_target_budget() -> None:
    req = SimulateRequest(
        observation_time_s=0.2,
        sample_rate_hz=20,
        roi_w=32,
        roi_h=32,
        seed=9,
    )
    params, _scenario = params_from_request(req)
    assert 1.0 <= params.target.target_range_m <= 10.0
    assert np.isclose(params.target.reference_range_m, params.target.target_range_m)
    assert np.isclose(params.target.target_area_m2, 0.025)
    assert np.isclose(params.optical.aperture_diameter_m, 0.025)
    assert np.isclose(params.optical.receiver_efficiency, 0.05)
    assert np.isclose(params.optical.filter_bandwidth_nm, 10.0)
    assert np.isclose(params.target.solar_irradiance_w_m2_nm, 0.000068)
    result = simulate_active_spad(params)
    assert result["mean_background_per_frame"] < 1e6, (
        f"default background is saturating the detector: {result['mean_background_per_frame']:.3g}"
    )
    assert result["dead_time_loss_ratio"] < 0.95
    assert not result["saturation_warning"], "default near-range scene should start outside detector saturation"


def main() -> None:
    tests = [
        test_frontend_projection_y_axis,
        test_scene_stray_background_has_stationary_floor_without_space_terms,
        test_background_spatial_map_stays_positive_after_large_gradients,
        test_recorded_trajectory_uses_circular_detector_fov,
        test_event_output_matches_sampled_frame_counts,
        test_event_timestamps_remain_inside_source_frame_after_jitter,
        test_pf32_timing_defaults_match_public_irf_specification,
        test_direct_python_defaults_match_pf32_detector,
        test_illumination_request_keeps_transmitter_sun_and_scene_stray_independent,
        test_default_transmitter_starts_in_near_range_single_photon_regime,
        test_laser_signal_rate_uses_transmitter_divergence_and_caps_intercepted_power,
        test_hybrid_signal_budget_is_sum_of_laser_and_solar_components,
        test_attitude_driven_laser_return_does_not_reuse_solar_direction,
        test_reported_snr_uses_photon_shot_noise,
        test_explicit_cuda_request_does_not_silently_fallback,
        test_custom_blade_uses_shape_points_not_center_blob,
        test_zero_rpm_keeps_blade_orientation_static,
        test_path_phase_drives_projected_blade_orientation,
        test_drone_four_propeller_centers_project_inside_detector,
        test_fast_drone_propellers_use_rotor_disk_projection,
        test_response_carries_backend_expected_signal_map,
        test_default_python_scene_uses_near_range_target_budget,
    ]
    for test in tests:
        test()
    print("backend projection checks passed")


if __name__ == "__main__":
    main()
