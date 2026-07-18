from __future__ import annotations

import sys
import inspect
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.convert import params_from_request, result_to_response, result_to_summary_response
from backend.exporters import generate_tdc_frame_cube_from_counts
from backend.models import SimulateRequest
from sim.active_imaging_sim import (
    _event_stream_from_counts,
    _frontend_drone_world_points,
    _frontend_blade_world_points,
    _background_series,
    _project_frontend_blade_local_points_to_pixels,
    _project_world_point_to_pixel,
    _shot_noise_snr_db,
    _shape_signal_distribution_cube,
    simulate_active_spad,
)
from sim.background import background_spatial_map
from sim import detector as detector_module
from sim.config import SimParams
from sim.imaging import _apply_separable_blur
from sim.physics import laser_target_detected_rate_cps, solar_environment_detected_rate_cps_per_pixel


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
    params = params_from_request(_base_request(**overrides))
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
    bg = _background_series(t, phase, los, params)
    assert np.all(bg > 0)
    assert float(np.mean(bg)) >= 7.0


def test_solar_environment_background_uses_detector_optics_without_manual_stray() -> None:
    common = {
        "scene_stray_rate": 0,
        "dark_count_rate": 0,
        "solar_irradiance": 1.35,
        "wavelength_nm": 780.0,
        "filter_bandwidth_nm": 10.0,
        "aperture_diameter_m": 0.025,
        "receiver_efficiency": 0.05,
        "quantum_efficiency": 0.07,
        "detector_fov_urad": FOV_URAD_50_DEG,
        "roi_w": 32,
        "roi_h": 32,
    }
    t = np.linspace(0, 0.04, 4)
    phase = np.zeros_like(t)
    los = np.tile(np.array([[0.0, 0.0, 1.0]]), (4, 1))
    narrow = _background_series(t, phase, los, _params(**common))
    wide = _background_series(t, phase, los, _params(**{**common, "filter_bandwidth_nm": 50.0}))
    no_sun = _background_series(t, phase, los, _params(**{**common, "solar_irradiance": 0.0}))

    assert float(np.mean(narrow)) > 100.0, "solar-driven ambient photons were not entering the detector background"
    assert float(np.mean(wide)) > float(np.mean(narrow)) * 4.5, "background should scale with filter bandwidth"
    assert np.allclose(no_sun, 0.0), "solar environment background must vanish when solar irradiance is zero"


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
    params = params_from_request(
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
    params = params_from_request(
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


def test_daylight_wide_fov_solar_background_is_not_stray_rejected() -> None:
    assert SimParams().optical.stray_light_rejection_ratio == 1.0
    pixel_ifov_urad = FOV_URAD_50_DEG / 32
    bg_rate = solar_environment_detected_rate_cps_per_pixel(
        irradiance_w_m2_nm=1.352,
        filter_bandwidth_nm=10.0,
        wavelength_nm=780.0,
        aperture_diameter_m=0.025,
        receiver_efficiency=0.05,
        quantum_efficiency=0.07,
        pixel_ifov_urad=pixel_ifov_urad,
        stray_light_rejection_ratio=1.0,
        visibility_km=23.0,
    )
    assert bg_rate > 1e9


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


def test_manual_sbr_background_overrides_solar_environment_noise() -> None:
    common = {
        "observation_time_s": 0.04,
        "sample_rate_hz": 50,
        "dark_count_rate": 0,
        "background_noise_mode": "manual_sbr",
        "manual_signal_background_ratio": 4.0,
        "laser_mode": "cw",
        "laser_average_power_w": 1e-6,
        "transmitter_divergence_mrad": 10.0,
        "illumination_mode": "laser",
    }

    bright = simulate_active_spad(_params(**{**common, "solar_irradiance": 10.0, "scene_stray_rate": 100000.0}))
    dark = simulate_active_spad(_params(**{**common, "solar_irradiance": 0.0, "scene_stray_rate": 0.0}))

    assert bright["total_signal_photons"] > 0
    assert np.isclose(
        bright["total_background_photons"],
        bright["total_signal_photons"] / 4.0,
        rtol=1e-6,
    )
    assert np.isclose(
        dark["total_background_photons"],
        dark["total_signal_photons"] / 4.0,
        rtol=1e-6,
    )
    assert np.isclose(bright["total_background_photons"], dark["total_background_photons"], rtol=1e-6)
    assert "manual_sbr" in bright["background_components"]


def test_manual_sbr_keeps_noise_when_target_is_out_of_fov() -> None:
    params = _params(
        observation_time_s=0.04,
        sample_rate_hz=50,
        target_position_x_m=10.0,
        target_position_y_m=1.0,
        target_position_z_m=1.5,
        dark_count_rate=0,
        background_noise_mode="manual_sbr",
        manual_signal_background_ratio=0.01,
        laser_mode="cw",
        laser_average_power_w=1e-6,
        transmitter_divergence_mrad=10.0,
        illumination_mode="laser",
    )
    result = simulate_active_spad(params)

    assert result["mean_in_fov_ratio"] == 0.0
    assert result["total_signal_photons"] == 0.0
    assert result["total_background_photons"] > 0.0
    assert result["observed_total_counts"] > 0


def test_manual_sbr_background_does_not_follow_target_frequency() -> None:
    params = _params(
        observation_time_s=1.0,
        sample_rate_hz=1000,
        dark_count_rate=0,
        background_noise_mode="manual_sbr",
        manual_signal_background_ratio=0.01,
        laser_mode="cw",
        laser_average_power_w=1e-6,
        transmitter_divergence_mrad=10.0,
        illumination_mode="laser",
        body_shape="blade_strip",
        spin_hz=200.0,
    )
    result = simulate_active_spad(params)
    background = np.asarray(result["background_components"]["manual_sbr"], dtype=np.float64)

    assert background.size == 1000
    assert float(np.mean(background)) > 0.0
    freqs = np.fft.rfftfreq(background.size, d=1.0 / params.sample_rate_hz)
    amplitudes = np.abs(np.fft.rfft(background - np.mean(background)))
    target_bin = int(np.argmin(np.abs(freqs - 200.0)))
    non_dc_max = float(np.max(amplitudes[1:]))
    assert float(amplitudes[target_bin]) < max(1e-9, non_dc_max * 0.2)


def test_tdc_background_noise_uses_random_tof_bins() -> None:
    counts = np.ones((20, 4, 4), dtype=np.uint16)
    signal = np.zeros_like(counts, dtype=np.float64)
    background = np.ones_like(counts, dtype=np.float64)
    dark = np.zeros_like(counts, dtype=np.float64)
    tdc = generate_tdc_frame_cube_from_counts(
        counts=counts,
        tdc_bin_width_ns=1.0,
        tdc_max_count=100,
        timing_jitter_ns=0.0,
        signal_cube=signal,
        bg_expected_cube=background,
        dark_expected_cube=dark,
        truth_range_series=np.full(counts.shape[0], 10.0, dtype=np.float64),
        rng=np.random.default_rng(123),
        empty_pixel_value=102,
    )

    truth_bin = int(round((2.0 * 10.0 / 299_792_458.0) / 1e-9))
    active_bins = tdc[tdc != 102]
    assert active_bins.size == counts.size
    assert len(np.unique(active_bins)) > 10
    assert not np.all(active_bins == truth_bin)


def test_tdc_static_range_fallback_uses_physical_tof() -> None:
    counts = np.ones((2, 1, 1), dtype=np.uint16)
    signal = np.ones_like(counts, dtype=np.float32)
    zeros = np.zeros_like(counts, dtype=np.float32)
    tdc = generate_tdc_frame_cube_from_counts(
        counts=counts,
        tdc_bin_width_ns=1.0,
        tdc_max_count=100,
        timing_jitter_ns=0.0,
        signal_cube=signal,
        bg_expected_cube=zeros,
        dark_expected_cube=zeros,
        truth_range_series=None,
        fallback_range_m=10.0,
        rng=np.random.default_rng(123),
        empty_pixel_value=0,
    )
    expected_bin = int(round((2.0 * 10.0 / 299_792_458.0) / 1e-9))
    assert np.all(tdc == expected_bin)


def test_summary_only_uses_aggregated_preview_without_frame_cube() -> None:
    common = {
        "observation_time_s": 0.2,
        "sample_rate_hz": 100,
        "save_truth_series": False,
        "dark_count_rate": 0,
        "background_noise_mode": "manual_sbr",
        "manual_signal_background_ratio": 5.0,
        "laser_mode": "cw",
        "laser_average_power_w": 1e-6,
        "transmitter_divergence_mrad": 10.0,
        "illumination_mode": "laser",
        "solar_irradiance": 10.0,
        "scene_stray_rate": 100000.0,
    }

    full = simulate_active_spad(_params(**common))
    fast_params = _params(**common)
    fast_params.summary_only = True
    fast = simulate_active_spad(fast_params)

    assert full["counts"].shape[0] == full["n_frames"]
    assert fast["counts"].shape[0] == 0
    assert fast["preview_counts"].shape == (fast["roi_h"], fast["roi_w"])
    assert fast["observed_total_counts"] == int(np.sum(fast["preview_counts"]))
    assert np.isclose(fast["total_signal_photons"], full["total_signal_photons"], rtol=1e-10)
    assert np.isclose(fast["total_background_photons"], fast["total_signal_photons"] / 5.0, rtol=1e-6)
    assert np.isclose(fast["total_background_photons"], full["total_background_photons"], rtol=1e-10)


def test_summary_only_preserves_per_frame_count_cap() -> None:
    params = _params(
        observation_time_s=0.04,
        sample_rate_hz=50,
        max_count_per_frame=2,
        dark_count_rate=1e12,
        scene_stray_rate=0,
        illumination_mode="laser",
        laser_average_power_w=0,
        save_truth_series=False,
    )
    params.summary_only = True
    result = simulate_active_spad(params)
    max_possible = result["n_frames"] * result["roi_h"] * result["roi_w"] * 2
    assert result["observed_total_counts"] <= max_possible
    assert result["saturation_warning"]


def test_short_observation_produces_one_frame() -> None:
    params = _params(observation_time_s=0.01, sample_rate_hz=1, save_truth_series=False)
    params.summary_only = True
    result = simulate_active_spad(params)
    assert result["n_frames"] == 1


def test_trajectory_extension_is_revalidated_against_budget() -> None:
    request = _base_request(
        observation_time_s=0.1,
        sample_rate_hz=10,
        target_trajectory_times_s=[0.0, 30_000.0],
        target_trajectory_x_m=[0.0, 0.0],
        target_trajectory_y_m=[1.0, 1.0],
        target_trajectory_z_m=[1.5, 1.5],
    )
    try:
        params_from_request(request)
        assert False, "trajectory-expanded simulation should exceed the backend frame budget"
    except ValueError as exc:
        assert "after trajectory conversion" in str(exc)


def test_separable_blur_uses_vectorized_hot_path() -> None:
    source = inspect.getsource(_apply_separable_blur)
    assert "apply_along_axis" not in source


def test_separable_blur_matches_reference_convolution() -> None:
    cube = np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5)

    def kernel(sigma: float) -> np.ndarray:
        radius = int(np.ceil(3.0 * sigma))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (x / sigma) ** 2)
        return weights / np.sum(weights)

    expected = cube.copy()
    kernel_x = kernel(0.8)
    padded_x = np.pad(expected, ((0, 0), (0, 0), (len(kernel_x) // 2, len(kernel_x) // 2)), mode="edge")
    expected = np.apply_along_axis(lambda row: np.convolve(row, kernel_x, mode="valid"), 2, padded_x)
    kernel_y = kernel(1.1)
    padded_y = np.pad(expected, ((0, 0), (len(kernel_y) // 2, len(kernel_y) // 2), (0, 0)), mode="edge")
    expected = np.apply_along_axis(lambda col: np.convolve(col, kernel_y, mode="valid"), 1, padded_y)

    actual = _apply_separable_blur(cube, 0.8, 1.1)
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


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


def test_custom_blade_vector_projection_matches_scalar_projection() -> None:
    params = _params(
        detector_yaw_deg=3.0,
        detector_pitch_deg=-2.0,
        target_pitch_deg=17.0,
    )
    target_pos = (0.2, 1.4, 3.5)
    local_x = np.array([-0.18, -0.05, 0.04, 0.21], dtype=np.float64)
    local_z = np.array([0.0, 0.03, -0.04, 0.02], dtype=np.float64)
    angle = 0.73
    pitch = np.deg2rad(17.0)
    cols, rows, valid = _project_frontend_blade_local_points_to_pixels(
        target_pos,
        local_x,
        local_z,
        angle,
        pitch,
        params,
        32,
        32,
    )

    scalar = [
        _project_world_point_to_pixel(point, params, 32, 32)
        for point in _frontend_blade_world_points(target_pos, list(zip(local_x, local_z)), angle, pitch)
    ]
    assert np.array_equal(valid, np.array([item is not None for item in scalar]))
    for index, projected in enumerate(scalar):
        if projected is None:
            continue
        assert np.isclose(cols[index], projected[0])
        assert np.isclose(rows[index], projected[1])


def test_blade_pitch_changes_backend_return_strength() -> None:
    common = {
        "body_shape": "blade_strip",
        "target_length_m": 0.45,
        "target_width_m": 0.05,
        "spin_hz": 0,
        "target_trajectory_times_s": [0.0, 0.1],
        "target_trajectory_x_m": [0.0, 0.0],
        "target_trajectory_y_m": [1.0, 1.0],
        "target_trajectory_z_m": [1.5, 1.5],
        "target_trajectory_phase_rad": [0.0, 0.0],
        "custom_shape_x": [-0.45, -0.2, 0.2, 0.45],
        "custom_shape_y": [-0.2, 0.2, -0.2, 0.2],
        "custom_shape_intensity": [1.0, 1.0, 1.0, 1.0],
        "custom_shape_aspect_ratio": 4.0,
    }
    low_pitch = _shape_cube(_params(**common, target_trajectory_pitch_deg=[0.0, 0.0]))
    high_pitch = _shape_cube(_params(**common, target_trajectory_pitch_deg=[70.0, 70.0]))
    assert abs(float(np.sum(high_pitch)) - float(np.sum(low_pitch))) > 0.2 * float(np.sum(low_pitch))


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
    params = params_from_request(req)
    result = simulate_active_spad(params)
    response = result_to_response(result, None)
    assert response.expected_signal_map_encoded is not None, "backend response missing expected signal map"
    assert result["expected_signal_map"].shape == (32, 32)
    assert float(np.sum(result["expected_signal_map"])) > 0


def test_recorded_rpm_keyframes_report_truth_frequency_series() -> None:
    req = _base_request(
        observation_time_s=2.5,
        sample_rate_hz=2,
        body_shape="blade_strip",
        target_length_m=0.45,
        target_width_m=0.05,
        spin_hz=0,
        target_trajectory_times_s=[0.0, 1.0, 2.0],
        target_trajectory_x_m=[0.0, 0.0, 0.0],
        target_trajectory_y_m=[1.0, 1.0, 1.0],
        target_trajectory_z_m=[1.5, 1.5, 1.5],
        target_trajectory_propeller_rpm1=[18000.0, 20000.0, 18000.0],
        target_trajectory_propeller_rpm2=[18000.0, 20000.0, 18000.0],
        target_trajectory_propeller_rpm3=[18000.0, 20000.0, 18000.0],
        target_trajectory_propeller_rpm4=[18000.0, 20000.0, 18000.0],
    )
    params = params_from_request(req)
    result = simulate_active_spad(params)
    summary = result_to_summary_response(result, None)
    expected_hz = np.array([300.0, 316.6666667, 333.3333333, 316.6666667, 300.0], dtype=np.float64)

    assert np.allclose(result["truth_frequency_series_hz"], expected_hz, rtol=0, atol=1e-3)
    assert np.isclose(result["truth_freq_hz"], float(np.mean(expected_hz)), atol=1e-3)
    assert summary.truth_frequency_series_hz is not None
    assert np.isclose(summary.truth_freq_hz, float(np.mean(expected_hz)), atol=1e-3)
    assert np.allclose(summary.truth_frequency_series_hz, expected_hz, rtol=0, atol=1e-3)


def test_drone_rpm_keyframes_report_four_propeller_frequency_series() -> None:
    req = _base_request(
        observation_time_s=2.5,
        sample_rate_hz=2,
        body_shape="drone_quad",
        target_length_m=0.30,
        target_width_m=0.30,
        target_height_m=0.08,
        propeller_diameter_m=0.12,
        spin_hz=0,
        target_trajectory_times_s=[0.0, 1.0, 2.0],
        target_trajectory_x_m=[0.0, 0.0, 0.0],
        target_trajectory_y_m=[1.0, 1.0, 1.0],
        target_trajectory_z_m=[1.5, 1.5, 1.5],
        target_trajectory_propeller_rpm1=[18000.0, 18000.0, 24000.0],
        target_trajectory_propeller_rpm2=[12000.0, 15000.0, 18000.0],
        target_trajectory_propeller_rpm3=[9000.0, 9000.0, 9000.0],
        target_trajectory_propeller_rpm4=[24000.0, 18000.0, 12000.0],
    )
    params = params_from_request(req)
    result = simulate_active_spad(params)
    summary = result_to_summary_response(result, None)
    expected = np.array(
        [
            [300.0, 200.0, 150.0, 400.0],
            [300.0, 225.0, 150.0, 350.0],
            [300.0, 250.0, 150.0, 300.0],
            [350.0, 275.0, 150.0, 250.0],
            [400.0, 300.0, 150.0, 200.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(result["truth_propeller_frequency_series_hz"], expected, rtol=0, atol=1e-3)
    assert summary.truth_propeller_frequency_series_hz is not None
    assert np.allclose(summary.truth_propeller_frequency_series_hz, expected, rtol=0, atol=1e-3)
    assert np.allclose(result["truth_frequency_series_hz"], np.mean(expected, axis=1), rtol=0, atol=1e-3)


def test_default_python_scene_uses_near_range_target_budget() -> None:
    req = SimulateRequest(
        observation_time_s=0.2,
        sample_rate_hz=20,
        roi_w=32,
        roi_h=32,
        seed=9,
    )
    params = params_from_request(req)
    assert 1.0 <= params.target.target_range_m <= 10.0
    assert np.isclose(params.target.reference_range_m, params.target.target_range_m)
    assert np.isclose(params.target.target_area_m2, 0.025)
    assert np.isclose(params.optical.aperture_diameter_m, 0.025)
    assert np.isclose(params.optical.receiver_efficiency, 0.05)
    assert np.isclose(params.optical.filter_bandwidth_nm, 10.0)
    assert np.isclose(params.target.solar_irradiance_w_m2_nm, 1.35)
    result = simulate_active_spad(params)
    assert 1e10 < result["mean_background_per_frame"] < 1e12, (
        f"default daylight environment background is outside expected range: {result['mean_background_per_frame']:.3g}"
    )
    assert result["saturation_warning"], "default near-range daylight scene should expose detector saturation risk"


def main() -> None:
    tests = [
        test_frontend_projection_y_axis,
        test_scene_stray_background_has_stationary_floor_without_space_terms,
        test_solar_environment_background_uses_detector_optics_without_manual_stray,
        test_background_spatial_map_stays_positive_after_large_gradients,
        test_recorded_trajectory_uses_circular_detector_fov,
        test_event_output_matches_sampled_frame_counts,
        test_event_timestamps_remain_inside_source_frame_after_jitter,
        test_pf32_timing_defaults_match_public_irf_specification,
        test_direct_python_defaults_match_pf32_detector,
        test_illumination_request_keeps_transmitter_sun_and_scene_stray_independent,
        test_default_transmitter_starts_in_near_range_single_photon_regime,
        test_laser_signal_rate_uses_transmitter_divergence_and_caps_intercepted_power,
        test_daylight_wide_fov_solar_background_is_not_stray_rejected,
        test_hybrid_signal_budget_is_sum_of_laser_and_solar_components,
        test_manual_sbr_background_overrides_solar_environment_noise,
        test_manual_sbr_keeps_noise_when_target_is_out_of_fov,
        test_manual_sbr_background_does_not_follow_target_frequency,
        test_tdc_background_noise_uses_random_tof_bins,
        test_tdc_static_range_fallback_uses_physical_tof,
        test_summary_only_uses_aggregated_preview_without_frame_cube,
        test_summary_only_preserves_per_frame_count_cap,
        test_short_observation_produces_one_frame,
        test_trajectory_extension_is_revalidated_against_budget,
        test_separable_blur_uses_vectorized_hot_path,
        test_separable_blur_matches_reference_convolution,
        test_attitude_driven_laser_return_does_not_reuse_solar_direction,
        test_reported_snr_uses_photon_shot_noise,
        test_explicit_cuda_request_does_not_silently_fallback,
        test_custom_blade_uses_shape_points_not_center_blob,
        test_custom_blade_vector_projection_matches_scalar_projection,
        test_blade_pitch_changes_backend_return_strength,
        test_zero_rpm_keeps_blade_orientation_static,
        test_path_phase_drives_projected_blade_orientation,
        test_drone_four_propeller_centers_project_inside_detector,
        test_fast_drone_propellers_use_rotor_disk_projection,
        test_response_carries_backend_expected_signal_map,
        test_recorded_rpm_keyframes_report_truth_frequency_series,
        test_drone_rpm_keyframes_report_four_propeller_frequency_series,
        test_default_python_scene_uses_near_range_target_budget,
    ]
    for test in tests:
        test()
    print("backend projection checks passed")


if __name__ == "__main__":
    main()
