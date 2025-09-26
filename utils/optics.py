# Copyright (c) 2025, ETH Zurich


import typing
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def patterned_wavefront(
        intensity: typing.Iterable[float],
        phase: typing.Iterable[float],
        visibility: typing.Iterable[float],
        intensity_unit = None,
        phase_unit = "rad",
        visibility_unit = "1") -> xr.Dataset:

    intensity = xr.DataArray(
        data=intensity,
        dims=("column",) if len(intensity.shape) == 1 else ("row", "column"))
    phase = xr.DataArray(
        data=phase,
        dims=("column",) if len(phase.shape) == 1 else ("row", "column"))
    visibility = xr.DataArray(
        data=visibility,
        dims=("column",) if len(visibility.shape) == 1 else ("row", "column"))

    patterned_wavefront = xr.Dataset(dict(
        intensity=intensity,
        phase=phase,
        visibility=visibility))

    patterned_wavefront.intensity.attrs["units"] = intensity_unit
    patterned_wavefront.phase.attrs["units"] = phase_unit
    patterned_wavefront.visibility.attrs["units"] = visibility_unit

    return patterned_wavefront


def flat_patterned_wavefront(
        intensity: float,
        phase: float,
        visibility: float,
        cols: int,
        rows: typing.Union[int, None] = None,
        **kwargs) -> xr.Dataset:

    dims = cols if rows is None else (rows, cols)
    #print(dims)
    #print((intensity * np.ones(dims)).shape)

    return patterned_wavefront(
        intensity=intensity * np.ones(dims, dtype=np.float32),
        phase=phase * np.ones(dims, dtype=np.float32),
        visibility=visibility * np.ones(dims, dtype=np.float32),
        **kwargs)


def refraction_angle_to_pattern_phase_shift(
        refraction_angle: xr.DataArray,
        sample_analyser_distance: float,
        pattern_pitch: float,
        name: str = "pattern phase shift",
        unit: str = "rad") -> xr.DataArray:

    pattern_phase_shift = 2 * np.pi * sample_analyser_distance * refraction_angle / pattern_pitch

    if name is not None:
        pattern_phase_shift.name = name

    if unit is not None:
        pattern_phase_shift.attrs["units"] = unit

    return pattern_phase_shift


def pattern_phase_shift_to_refraction_angle(
        pattern_phase_shift: xr.DataArray,
        sample_analyser_distance: float,
        pattern_pitch: float,
        name: str = "refraction angle",
        unit: str = "rad") -> xr.DataArray:

    refraction_angle = pattern_phase_shift * pattern_pitch / (2 * np.pi * sample_analyser_distance)

    if name is not None:
        refraction_angle.name = name

    if unit is not None:
        refraction_angle.attrs["units"] = unit

    return refraction_angle


def propagate_patterned_wavefront_through_sample(
        patterned_wavefront: xr.Dataset,
        sample_attenuation: xr.DataArray,
        sample_pattern_shift: xr.DataArray,
        sample_diffusion: xr.DataArray) -> xr.Dataset:

    # Here, surprisingly, the order matters. Putting the sample data
    # first ensures that the order of the dimesions in the output
    # is the same as in the input.
    intensity = sample_attenuation * patterned_wavefront.intensity
    visibility = sample_diffusion * patterned_wavefront.visibility
    phase = sample_pattern_shift + patterned_wavefront.phase

    intensity.attrs["units"] = patterned_wavefront.intensity.attrs.get("units")
    visibility.attrs["units"] = patterned_wavefront.visibility.attrs.get("units")
    phase.attrs["units"] = patterned_wavefront.phase.attrs.get("units")

    propagated_patterned_wavefront = xr.Dataset(dict(
        intensity=intensity,
        phase=phase,
        visibility=visibility))

    return propagated_patterned_wavefront


def relate_to_reference_patterned_wavefront(
        reference_patterned_wavefront: xr.Dataset,
        patterned_wavefront_after_sample: xr.Dataset) -> xr.Dataset:

    attenuation = -np.log(
        patterned_wavefront_after_sample.intensity / reference_patterned_wavefront.intensity)
    diffusion = -np.log(
        patterned_wavefront_after_sample.visibility / reference_patterned_wavefront.visibility)

    pattern_phase_shift = patterned_wavefront_after_sample.phase - reference_patterned_wavefront.phase
    pattern_phase_shift.name = "pattern phase shift"
    pattern_phase_shift.attrs["units"] = "rad"

    relative_change_in_wavefront = xr.Dataset(dict(
        attenuation=attenuation,
        diffusion=diffusion,
        pattern_phase_shift=pattern_phase_shift))

    return relative_change_in_wavefront