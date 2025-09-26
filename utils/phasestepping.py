# Copyright (c) 2025, ETH Zurich

import typing
import numpy as np
import scipy.optimize
import xarray as xr

from . import ut

def phase_stepping_array(
        phasesteps: typing.Iterable[float]) -> xr.DataArray:

    x = xr.DataArray(
        phasesteps,
        dims=("step"),
        name="phase step",
        attrs=dict(units="angle"))

    return x


def equidistant_phase_stepping_array(n: int) -> xr.DataArray:
    return phase_stepping_array(
        np.linspace(0, 2*np.pi, n, endpoint=False, dtype = np.float32))

def phase_stepping_model_by_mean(
        phase_steps: xr.DataArray,
        wavefront: xr.Dataset,
        name: str = "intensity",
        unit: str = None,
        norm: float = None) -> xr.DataArray:
    
    if norm is None:
        norm = phase_steps.size

    # i_max = (wavefront.intensity / norm)
    # i_min = i_max * (1 - wavefront.visibility) / (1 + wavefront.visibility)
    # i_avg = (i_max + i_min) / 2
    i_avg = wavefront.intensity
    intensity_steps = (i_avg / norm) * (1 + wavefront.visibility * \
                               np.cos(wavefront.phase + phase_steps))
    
    
    intensity_steps.name = name

    if (unit is None) and ("units" in wavefront.intensity.attrs.keys()):
        intensity_steps.attrs["units"] = wavefront.intensity.attrs["units"]

    return intensity_steps

def fixphase(d, threshold = np.pi):
    """Correct wrapping that occurs when subtracting two phases.
    """
    d[d < -threshold] += 2*np.pi
    d[d > threshold] -= 2*np.pi

    return d

def signal_retrieval(d, axis):
    """Retrieve a interferometry signal from a multi-dimensional dataset.
    """
    dfft = np.fft.rfft(d, axis=axis)
    dfft_abs = np.abs(dfft)

    dabs = dfft_abs.take(0, axis=axis)
    dphase = np.angle(dfft).take(1, axis=axis)
    dvis = 2 *  dfft_abs.take(1, axis=axis) / dabs

    return dabs.astype(np.float32), dphase.astype(np.float32), dvis.astype(np.float32)


def signal_retrieval_da(d: xr.DataArray) -> xr.Dataset:
    dabs, dphase, dvis = signal_retrieval(
        d.values.astype(np.float32),
        axis=d.dims.index("step"))

    dims = tuple(x for x in d.dims if x != "step")

    retrieved_signal = xr.Dataset(dict(
        # intensity is normalized to the amplitude
        intensity=(dims, dabs * 2),
        phase=(dims, dphase),
        visibility=(dims, dvis)),
        coords=d.coords)

    if "units" in d.attrs:
        retrieved_signal.intensity.attrs["units"] = d.attrs["units"]
    retrieved_signal.phase.attrs["units"] = "rad"

    return retrieved_signal


def signal_retrieval_da_by_mean(d: xr.DataArray) -> xr.Dataset:
    dabs, dphase, dvis = signal_retrieval(
        d.values.astype(np.float32),
        axis=d.dims.index("step"))

    dims = tuple(x for x in d.dims if x != "step")

    retrieved_signal = xr.Dataset(dict(
        # intensity is normalized to the amplitude
        intensity=(dims, dabs),
        phase=(dims, dphase),
        visibility=(dims, dvis)),
        coords=d.coords)

    if "units" in d.attrs:
        retrieved_signal.intensity.attrs["units"] = d.attrs["units"]
    retrieved_signal.phase.attrs["units"] = "rad"

    return retrieved_signal
