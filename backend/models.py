"""Pydantic request and response models for the SPAD simulator API."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


SimulationTier = Literal["baseline_empirical", "physics_informed"]
OutputMode = Literal["frame", "event"]
LightcurveMode = Literal["analytic_modulation", "attitude_driven"]
BodyShape = Literal["sphere", "plate", "blade_strip", "drone_quad"]
DetectorPreset = Literal["custom", "pf32"]
ComputeBackend = Literal["auto", "cpu", "cuda"]
IlluminationMode = Literal["laser", "solar", "laser_plus_solar"]
LaserMode = Literal["pulsed", "cw"]

MAX_BACKEND_FRAMES = 200_000
MAX_BACKEND_ROI_PIXELS = 16_384
MAX_BACKEND_SAMPLES = 204_800_000
MAX_BACKEND_TRAJECTORY_POINTS = 50_000


class SimulateRequest(BaseModel):
    """Simulation request parameters."""

    scenario: str | None = None
    detector_preset: DetectorPreset | None = None

    observation_time_s: float | None = Field(default=None, gt=0)
    sample_rate_hz: float | None = Field(default=None, gt=0)
    seed: int | None = None
    compute_backend: ComputeBackend | None = None
    simulation_tier: SimulationTier | None = None
    output_mode: OutputMode | None = None
    lightcurve_mode: LightcurveMode | None = None
    save_truth_series: bool | None = None

    target_range_m: float | None = Field(default=None, gt=0)
    target_radial_velocity_mps: float | None = None
    target_radial_accel_mps2: float | None = None
    target_area_m2: float | None = Field(default=None, gt=0)
    target_length_m: float | None = Field(default=None, gt=0)
    target_width_m: float | None = Field(default=None, gt=0)
    target_height_m: float | None = Field(default=None, gt=0)
    propeller_diameter_m: float | None = Field(default=None, gt=0)
    target_reflectivity: float | None = Field(default=None, ge=0, le=1)
    propeller_reflectivity: float | None = Field(default=None, ge=0, le=1)
    solar_irradiance: float | None = Field(default=None, ge=0)
    phase_function_scale: float | None = Field(default=None, ge=0)
    specular_fraction: float | None = Field(default=None, ge=0, le=1)
    modulation_depth: float | None = Field(default=None, ge=0, le=1)
    tumbling_hz: float | None = Field(default=None, ge=0)
    spin_hz: float | None = Field(default=None, ge=0)
    precession_hz: float | None = Field(default=None, ge=0)
    body_shape: BodyShape | None = None
    outage_fraction: float | None = Field(default=None, ge=0, le=1)
    glint_probability: float | None = Field(default=None, ge=0, le=1)
    glint_gain: float | None = Field(default=None, ge=1)
    target_position_x_m: float | None = None
    target_position_y_m: float | None = None
    target_position_z_m: float | None = None
    target_yaw_deg: float | None = None
    target_pitch_deg: float | None = None
    target_roll_deg: float | None = None
    target_trajectory_times_s: list[float] | None = None
    target_trajectory_x_m: list[float] | None = None
    target_trajectory_y_m: list[float] | None = None
    target_trajectory_z_m: list[float] | None = None
    target_trajectory_yaw_deg: list[float] | None = None
    target_trajectory_pitch_deg: list[float] | None = None
    target_trajectory_roll_deg: list[float] | None = None
    target_trajectory_phase_rad: list[float] | None = None
    target_trajectory_propeller_rpm1: list[float] | None = None
    target_trajectory_propeller_rpm2: list[float] | None = None
    target_trajectory_propeller_rpm3: list[float] | None = None
    target_trajectory_propeller_rpm4: list[float] | None = None
    target_trajectory_propeller_phase1_rad: list[float] | None = None
    target_trajectory_propeller_phase2_rad: list[float] | None = None
    target_trajectory_propeller_phase3_rad: list[float] | None = None
    target_trajectory_propeller_phase4_rad: list[float] | None = None
    custom_shape_x: list[float] | None = None
    custom_shape_y: list[float] | None = None
    custom_shape_intensity: list[float] | None = None
    custom_shape_aspect_ratio: float | None = Field(default=None, gt=0)
    detector_position_x_m: float | None = None
    detector_position_y_m: float | None = None
    detector_position_z_m: float | None = None
    detector_yaw_deg: float | None = None
    detector_pitch_deg: float | None = None

    scene_stray_rate: float | None = Field(default=None, ge=0)

    illumination_mode: IlluminationMode | None = None
    laser_mode: LaserMode | None = None
    laser_average_power_w: float | None = Field(default=None, ge=0)
    laser_pulse_energy_j: float | None = Field(default=None, ge=0)
    laser_repetition_frequency_hz: float | None = Field(default=None, gt=0)
    laser_pulse_width_ns: float | None = Field(default=None, ge=0)
    transmitter_divergence_mrad: float | None = Field(default=None, gt=0)

    aperture_diameter_m: float | None = Field(default=None, gt=0)
    receiver_efficiency: float | None = Field(default=None, ge=0, le=1)
    quantum_efficiency: float | None = Field(default=None, ge=0, le=1)
    wavelength_nm: float | None = Field(default=None, gt=0)
    filter_bandwidth_nm: float | None = Field(default=None, gt=0)
    detector_fov_urad: float | None = Field(default=None, gt=0)
    atmospheric_attenuation_enabled: bool | None = None
    atmospheric_visibility_km: float | None = Field(default=None, gt=0)

    dark_count_rate: float | None = Field(default=None, ge=0)
    dead_time_ns: float | None = Field(default=None, ge=0)
    max_count_per_frame: int | None = Field(default=None, ge=0)
    timing_jitter_ns: float | None = Field(default=None, ge=0)
    tdc_bin_width_ns: float | None = Field(default=None, ge=0)
    irf_fwhm_ps: float | None = Field(default=None, ge=0)
    max_count_rate_cps_per_pixel: float | None = Field(default=None, ge=0)
    detector_off_axis_urad_x: float | None = None
    detector_off_axis_urad_y: float | None = None

    roi_w: int | None = Field(default=None, ge=2)
    roi_h: int | None = Field(default=None, ge=2)
    pixel_pitch_um: float | None = Field(default=None, gt=0)
    fill_factor: float | None = Field(default=None, ge=0, le=1)
    microlens_gain: float | None = Field(default=None, ge=0)
    drift_pixels_per_s_x: float | None = None
    drift_pixels_per_s_y: float | None = None
    jitter_sigma_pixels: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_backend_budget(self) -> "SimulateRequest":
        observation_time_s = self.observation_time_s if self.observation_time_s is not None else 60.0
        sample_rate_hz = self.sample_rate_hz if self.sample_rate_hz is not None else 50.0
        roi_w = self.roi_w if self.roi_w is not None else 32
        roi_h = self.roi_h if self.roi_h is not None else 32

        n_frames = max(1, int(round(float(observation_time_s) * float(sample_rate_hz))))
        roi_pixels = int(roi_w) * int(roi_h)
        total_samples = n_frames * roi_pixels
        if n_frames > MAX_BACKEND_FRAMES:
            raise ValueError(f"simulation frame count exceeds backend limit ({n_frames} > {MAX_BACKEND_FRAMES})")
        if roi_pixels > MAX_BACKEND_ROI_PIXELS:
            raise ValueError(f"ROI pixel count exceeds backend limit ({roi_pixels} > {MAX_BACKEND_ROI_PIXELS})")
        if total_samples > MAX_BACKEND_SAMPLES:
            raise ValueError(f"simulation sample budget exceeds backend limit ({total_samples} > {MAX_BACKEND_SAMPLES})")

        required_trajectory_fields = (
            "target_trajectory_times_s",
            "target_trajectory_x_m",
            "target_trajectory_y_m",
            "target_trajectory_z_m",
        )
        provided_required = [
            field_name for field_name in required_trajectory_fields if getattr(self, field_name) is not None
        ]
        if provided_required:
            if len(provided_required) != len(required_trajectory_fields):
                raise ValueError("recorded trajectory requires times, x, y, and z arrays")
            lengths = {len(getattr(self, field_name) or []) for field_name in required_trajectory_fields}
            if len(lengths) != 1:
                raise ValueError("recorded trajectory times, x, y, and z arrays must have the same length")
            trajectory_len = lengths.pop()
            if trajectory_len < 2:
                raise ValueError("recorded trajectory requires at least two points")
            if trajectory_len > MAX_BACKEND_TRAJECTORY_POINTS:
                raise ValueError(
                    f"recorded trajectory exceeds backend limit ({trajectory_len} > {MAX_BACKEND_TRAJECTORY_POINTS})"
                )
            for field_name in (
                "target_trajectory_yaw_deg",
                "target_trajectory_pitch_deg",
                "target_trajectory_roll_deg",
                "target_trajectory_phase_rad",
                "target_trajectory_propeller_rpm1",
                "target_trajectory_propeller_rpm2",
                "target_trajectory_propeller_rpm3",
                "target_trajectory_propeller_rpm4",
                "target_trajectory_propeller_phase1_rad",
                "target_trajectory_propeller_phase2_rad",
                "target_trajectory_propeller_phase3_rad",
                "target_trajectory_propeller_phase4_rad",
            ):
                values = getattr(self, field_name)
                if values is not None and len(values) != trajectory_len:
                    raise ValueError(f"{field_name} length must match recorded trajectory length")

        custom_shape_lengths = [
            len(values)
            for values in (self.custom_shape_x, self.custom_shape_y, self.custom_shape_intensity)
            if values is not None
        ]
        if custom_shape_lengths and len(set(custom_shape_lengths)) != 1:
            raise ValueError("custom shape x, y, and intensity arrays must have the same length")
        if custom_shape_lengths and custom_shape_lengths[0] > 512:
            raise ValueError("custom shape arrays exceed backend limit (512)")

        return self


class DetectorSummary(BaseModel):
    """Summary of the active detector configuration."""

    preset: DetectorPreset
    preset_label: str
    array_rows: int
    array_cols: int
    pixel_pitch_um: float
    fill_factor: float
    microlens_gain: float
    detector_fov_urad: float
    pixel_ifov_urad: float
    tdc_bin_width_ns: float
    irf_fwhm_ps: float
    max_count_rate_cps_per_pixel: float
    max_count_per_frame: int
    quantum_efficiency: float
    receiver_efficiency: float
    assumptions: list[str] = Field(default_factory=list)


class SimulateResponse(BaseModel):
    """Simulation response used by the web frontend."""

    scenario_id: str | None = None
    scenario_name: str | None = None
    simulation_tier: SimulationTier
    output_mode: OutputMode
    lightcurve_mode: LightcurveMode
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    detector: DetectorSummary

    snr_db: float
    truth_freq_hz: float
    truth_pixel: int
    truth_row: int
    truth_col: int
    n_frames: int
    roi_h: int
    roi_w: int
    sample_rate_hz: float
    target_detected_rate_cps: float
    target_laser_detected_rate_cps: float
    target_solar_detected_rate_cps: float
    mean_signal_per_frame: float
    mean_background_per_frame: float
    mean_dark_per_frame: float
    total_signal_photons: float
    total_background_photons: float
    total_noise_photons: float
    pixel_ifov_urad: float
    fov_clipping_ratio: float
    dead_time_loss_ratio: float
    saturation_warning: bool = False
    visibility_ratio: float
    dropout_ratio: float
    harmonic_truth_strength: float
    event_count: int = 0

    counts_encoded: str = ""
    event_times_encoded: str | None = None
    event_pixels_encoded: str | None = None
    expected_signal_map_encoded: str | None = None
    truth_signal_series_encoded: str | None = None
    truth_bg_base_series_encoded: str | None = None
    truth_bg_total_series_encoded: str | None = None
    truth_visibility_series_encoded: str | None = None
    truth_cx_series_encoded: str | None = None
    truth_cy_series_encoded: str | None = None
    truth_projected_width_px_series_encoded: str | None = None
    truth_projected_height_px_series_encoded: str | None = None
    truth_range_series_encoded: str | None = None
    truth_glint_series_encoded: str | None = None
    modulation_series_encoded: str | None = None
    pde_map_encoded: str = ""
    dark_map_encoded: str = ""
    bg_spatial_map_encoded: str = ""


class SimulateSummaryResponse(BaseModel):
    """Lightweight response for frontend display of a backend-generated run."""

    scenario_id: str | None = None
    scenario_name: str | None = None
    simulation_tier: SimulationTier
    output_mode: OutputMode
    lightcurve_mode: LightcurveMode
    compute_backend: Literal["cpu", "cuda"]
    sample_backend: Literal["cpu", "cuda"]
    encoded_payload_omitted: bool = True

    n_frames: int
    roi_h: int
    roi_w: int
    sample_rate_hz: float
    snr_db: float
    observed_total_counts: int
    total_signal_photons: float
    total_background_photons: float
    total_noise_photons: float
    mean_signal_per_frame: float
    mean_background_per_frame: float
    mean_dark_per_frame: float
    fov_clipping_ratio: float = 0.0
    mean_in_fov_ratio: float = 1.0
    atmospheric_transmission_mean: float = 1.0
    dead_time_loss_ratio: float
    saturation_warning: bool = False
    visibility_ratio: float
    dropout_ratio: float
    target_detected_rate_cps: float
    target_laser_detected_rate_cps: float
    target_solar_detected_rate_cps: float
    truth_freq_hz: float
    truth_row: int
    truth_col: int
    preview_counts: list[list[int]] = Field(default_factory=list)
    expected_signal_map: list[list[float]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


JobStatus = Literal["queued", "running", "completed", "failed"]


class SimulateJobStatusResponse(BaseModel):
    """Status for a backend simulation job."""

    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    summary: SimulateSummaryResponse | None = None
    result: SimulateResponse | None = None
    error: str | None = None
    download_url: str | None = None
