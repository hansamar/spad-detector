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
    _frontend_drone_world_points,
    _background_series,
    _project_world_point_to_pixel,
    _shape_signal_distribution_cube,
    simulate_active_spad,
)


FOV_URAD_50_DEG = 50.0 * np.pi / 180.0 * 1e6


def _base_request(**overrides):
    data = {
        "detector_preset": "pf32_nominal",
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


def test_default_remote_scene_keeps_physical_budget_finite() -> None:
    req = SimulateRequest(
        detector_preset="pf32_nominal",
        observation_time_s=0.2,
        sample_rate_hz=20,
        roi_w=32,
        roi_h=32,
        body_shape="blade_strip",
        target_area_m2=0.025,
        seed=9,
    )
    params, _scenario = params_from_request(req)
    result = simulate_active_spad(params)
    assert result["atmospheric_transmission_mean"] > 0.05, (
        "default remote scene should not treat the full 400 km range as dense-atmosphere path"
    )
    assert result["mean_background_per_frame"] < 1e6, (
        f"default background is saturating the detector: {result['mean_background_per_frame']:.3g}"
    )
    assert result["dead_time_loss_ratio"] < 0.95


def main() -> None:
    tests = [
        test_frontend_projection_y_axis,
        test_scene_stray_background_has_stationary_floor_without_space_terms,
        test_custom_blade_uses_shape_points_not_center_blob,
        test_zero_rpm_keeps_blade_orientation_static,
        test_path_phase_drives_projected_blade_orientation,
        test_drone_four_propeller_centers_project_inside_detector,
        test_fast_drone_propellers_use_rotor_disk_projection,
        test_response_carries_backend_expected_signal_map,
        test_default_remote_scene_keeps_physical_budget_finite,
    ]
    for test in tests:
        test()
    print("backend projection checks passed")


if __name__ == "__main__":
    main()
