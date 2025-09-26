# Copyright (c) 2025, ETH Zurich


# here we would have things like interpolating nans from imageanalysis

import typing
import scipy.interpolate
import yaml
import numpy as np
import xarray as xr
import skimage.transform
import skimage.filters

# Set a noise seed
np.random.seed(1)


def volume_dataarray(data, voxel_size):
    is3d = len(data.shape) > 2

    return xr.DataArray(
        data,
        dims=["z", "y", "x"] if is3d else ["y", "x"],
        coords=[
            xr.DataArray(
                (np.arange(data.shape[i], dtype = np.float32) - (data.shape[i] - 1) / 2) * voxel_size,
                attrs=dict(units="m"))
            for i in range(len(data.shape))
        ],
    )

def volume_dataset(data: typing.Dict, voxel_size: float) -> xr.Dataset:
    phantom = xr.Dataset()
    for key in data:
        phantom[key] = volume_dataarray(data[key], voxel_size)
    else:
        return phantom


def add_poisson_noise(d: xr.DataArray) -> xr.DataArray:
    dnoise = d.copy()
    dnoise.data = np.random.poisson(dnoise.data)
    return dnoise
