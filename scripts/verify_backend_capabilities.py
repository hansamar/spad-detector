import os
from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["SPAD_SIM_OUTPUT_DIR"] = str(PROJECT_ROOT / "output")

from backend.main import app
from backend.convert import params_from_request
from backend.models import SimulateRequest
from sim.spectral import am0_solar_irradiance_w_m2_nm, pf32_pdp_fraction


def main() -> None:
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200

    capabilities = client.get("/api/capabilities")
    assert capabilities.status_code == 200
    payload = capabilities.json()
    assert payload["cpu_workers_default"] == 8
    assert "python_executable" in payload
    assert "torch_available" in payload
    assert "cuda_available" in payload
    assert "recommended_backend" in payload

    too_large = client.post(
        "/api/simulate/summary",
        json={
            "observation_time_s": 10_000,
            "sample_rate_hz": 100_000,
            "roi_w": 512,
            "roi_h": 512,
        },
    )
    assert too_large.status_code == 422

    hundred_k_params, _scenario = params_from_request(
        SimulateRequest(
            detector_preset="pf32",
            observation_time_s=2.0,
            sample_rate_hz=50_000,
            roi_w=32,
            roi_h=32,
        )
    )
    assert int(round(hundred_k_params.observation_time_s * hundred_k_params.sample_rate_hz)) == 100_000
    two_hundred_k_params, _scenario = params_from_request(
        SimulateRequest(
            detector_preset="pf32",
            observation_time_s=4.0,
            sample_rate_hz=50_000,
            roi_w=32,
            roi_h=32,
        )
    )
    assert int(round(two_hundred_k_params.observation_time_s * two_hundred_k_params.sample_rate_hz)) == 200_000

    bad_trajectory = client.post(
        "/api/simulate/summary",
        json={
            "observation_time_s": 0.2,
            "sample_rate_hz": 20,
            "target_trajectory_times_s": [0.0, 0.1, 0.2],
            "target_trajectory_x_m": [0.0, 0.1],
            "target_trajectory_y_m": [1.0, 1.1, 1.2],
            "target_trajectory_z_m": [8.0, 8.1, 8.2],
        },
    )
    assert bad_trajectory.status_code == 422

    spectral_params, _ = params_from_request(
        SimulateRequest(
            wavelength_nm=850,
            observation_time_s=0.2,
            sample_rate_hz=20,
        )
    )
    assert spectral_params.detector_preset == "pf32"
    assert np.isclose(spectral_params.optical.quantum_efficiency, float(pf32_pdp_fraction(850)))
    assert np.isclose(spectral_params.target.solar_irradiance_w_m2_nm, float(am0_solar_irradiance_w_m2_nm(850)))

    override_params, _ = params_from_request(
        SimulateRequest(
            wavelength_nm=850,
            quantum_efficiency=0.123,
            solar_irradiance=0.456,
            observation_time_s=0.2,
            sample_rate_hz=20,
        )
    )
    assert override_params.detector_preset == "pf32"
    assert np.isclose(override_params.optical.quantum_efficiency, 0.123)
    assert np.isclose(override_params.target.solar_irradiance_w_m2_nm, 0.456)

    scene_params, _ = params_from_request(
        SimulateRequest(
            target_position_x_m=10,
            target_position_y_m=20,
            target_position_z_m=100,
            detector_position_x_m=0,
            detector_position_y_m=10,
            detector_position_z_m=0,
            detector_yaw_deg=0,
            detector_pitch_deg=0,
            target_yaw_deg=17,
            target_roll_deg=23,
            observation_time_s=0.2,
            sample_rate_hz=20,
        )
    )
    assert np.isclose(scene_params.target.target_range_m, np.sqrt(10**2 + 10**2 + 100**2))
    assert scene_params.optical.detector_off_axis_urad_x > 0
    assert scene_params.optical.detector_off_axis_urad_y < 0
    assert np.isclose(scene_params.target.spin_axis_azimuth_deg, 17)
    assert np.isclose(scene_params.target.phase1, np.deg2rad(23))

    recorded_params, _ = params_from_request(
        SimulateRequest(
            detector_preset="pf32",
            observation_time_s=0.2,
            sample_rate_hz=20,
            target_trajectory_times_s=[0.0, 0.1, 0.25],
            target_trajectory_x_m=[0.0, 0.4, 0.8],
            target_trajectory_y_m=[1.4, 1.6, 1.8],
            target_trajectory_z_m=[8.0, 9.0, 10.0],
            target_trajectory_propeller_rpm1=[11000, 11200, 11400],
            target_trajectory_propeller_rpm2=[10900, 11100, 11300],
            target_trajectory_propeller_rpm3=[11300, 11500, 11700],
            target_trajectory_propeller_rpm4=[11200, 11400, 11600],
            detector_position_x_m=0,
            detector_position_y_m=1,
            detector_position_z_m=0,
            detector_yaw_deg=0,
            detector_pitch_deg=0,
        )
    )
    assert recorded_params.geometry.trajectory_name == "manual_recorded_flight"
    assert recorded_params.geometry.external_range_m is not None
    assert recorded_params.geometry.external_off_axis_x_urad is not None
    assert recorded_params.geometry.external_in_fov is not None
    assert recorded_params.observation_time_s > 0.25

    summary = client.post(
        "/api/simulate/summary",
        json={
            "detector_preset": "pf32",
            "observation_time_s": 0.2,
            "sample_rate_hz": 20,
            "roi_w": 4,
            "roi_h": 4,
            "body_shape": "blade_strip",
            "target_area_m2": 0.025,
            "solar_irradiance": 0.8,
            "scene_stray_rate": 2,
            "save_truth_series": True,
        },
    )
    assert summary.status_code == 200
    sim = summary.json()
    assert sim["n_frames"] == 4
    assert sim["roi_w"] == 4
    assert sim["roi_h"] == 4
    assert sim["compute_backend"] in {"cpu", "cuda"}
    assert sim["sample_backend"] in {"cpu", "cuda"}
    assert sim["encoded_payload_omitted"] is True
    assert "counts_encoded" not in sim
    assert len(sim["preview_counts"]) == 4
    assert len(sim["expected_signal_map"]) == 4
    assert len(sim["expected_signal_map"][0]) == 4
    assert sum(sum(row) for row in sim["expected_signal_map"]) > 0
    assert sim["total_noise_photons"] >= sim["total_background_photons"]
    assert sim["total_noise_photons"] > sim["total_background_photons"]
    assert sim["target_laser_detected_rate_cps"] > 0
    assert sim["target_solar_detected_rate_cps"] > 0
    assert np.isclose(
        sim["target_detected_rate_cps"],
        sim["target_laser_detected_rate_cps"] + sim["target_solar_detected_rate_cps"],
    )

    job = client.post(
        "/api/simulate/jobs",
        json={
            "detector_preset": "pf32",
            "observation_time_s": 0.2,
            "sample_rate_hz": 20,
            "roi_w": 4,
            "roi_h": 4,
            "body_shape": "blade_strip",
            "target_area_m2": 0.025,
            "solar_irradiance": 0.8,
            "scene_stray_rate": 2,
            "save_truth_series": True,
        },
    )
    assert job.status_code == 200
    job_id = job.json()["job_id"]
    status_payload = None
    for _ in range(40):
        status = client.get(f"/api/simulate/jobs/{job_id}")
        assert status.status_code == 200
        status_payload = status.json()
        if status_payload["status"] == "completed":
            break
        assert status_payload["status"] in {"queued", "running"}
        time.sleep(0.1)
    assert status_payload is not None
    assert status_payload["status"] == "completed"
    assert status_payload["summary"]["n_frames"] == 4
    assert sum(sum(row) for row in status_payload["summary"]["expected_signal_map"]) > 0
    assert status_payload["summary"]["total_noise_photons"] >= status_payload["summary"]["total_background_photons"]
    assert status_payload["result"] is None
    assert status_payload["download_url"].endswith("/download")

    download = client.get(status_payload["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/octet-stream"
    assert download.headers.get("content-disposition", "").endswith('.bin"')
    assert len(download.content) == status_payload["summary"]["n_frames"] * status_payload["summary"]["roi_h"] * status_payload["summary"]["roi_w"] * 2

    print("backend capabilities checks passed")


if __name__ == "__main__":
    main()
