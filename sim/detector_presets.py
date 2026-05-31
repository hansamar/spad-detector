from __future__ import annotations

from dataclasses import dataclass

from sim.config import SimParams
from sim.spectral import am0_solar_irradiance_w_m2_nm, pf32_pdp_fraction


@dataclass(frozen=True)
class DetectorPresetInfo:
    key: str
    label: str
    assumptions: list[str]


PRESET_INFO: dict[str, DetectorPresetInfo] = {
    "custom": DetectorPresetInfo(
        key="custom",
        label="Custom detector",
        assumptions=["User-defined detector settings are used without PF32 preset overrides."],
    ),
    "pf32": DetectorPresetInfo(
        key="pf32",
        label="PF32",
        assumptions=[
            "32x32 silicon SPAD array based on PF32 public datasheet figures.",
            "Fill factor and microlens gain use engineering approximations for active imaging studies.",
        ],
    ),
}


def apply_detector_preset(params: SimParams, preset: str | None) -> None:
    selected = preset or "pf32"
    if selected not in PRESET_INFO:
        raise ValueError(f"Unsupported detector preset: {selected}")
    params.detector_preset = selected
    if selected == "custom":
        return

    # PF32: public figures plus engineering approximations for active imaging studies.
    params.image.roi_w = 32
    params.image.roi_h = 32
    params.image.center_x = 15.5
    params.image.center_y = 15.5
    params.image.pixel_pitch_um = 50.0
    params.image.fill_factor = 0.015
    params.image.microlens_gain = 13.3
    params.optical.quantum_efficiency = float(pf32_pdp_fraction(params.optical.wavelength_nm))
    params.spad.dark_count_rate_cps = 100.0
    params.spad.timing_jitter_ns = 0.2 / 2.355
    params.spad.tdc_bin_width_ns = 0.055
    params.spad.irf_fwhm_ps = 200.0
    params.spad.max_count_rate_cps_per_pixel = 20e6
    params.spad.max_count_per_frame = 65535


def refresh_pf32_spectral_defaults(
    params: SimParams,
    *,
    update_quantum_efficiency: bool = True,
    update_solar_irradiance: bool = True,
) -> None:
    """Keep PF32 wavelength-dependent defaults coherent after request overrides."""
    if params.detector_preset != "pf32":
        return
    if update_quantum_efficiency:
        params.optical.quantum_efficiency = float(pf32_pdp_fraction(params.optical.wavelength_nm))
    if update_solar_irradiance:
        params.target.solar_irradiance_w_m2_nm = float(am0_solar_irradiance_w_m2_nm(params.optical.wavelength_nm))


def preset_summary(params: SimParams) -> DetectorPresetInfo:
    return PRESET_INFO.get(params.detector_preset, PRESET_INFO["custom"])
