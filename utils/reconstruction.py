# Copyright (c) 2025, ETH Zurich


# here the reconstruction goes, at least the basic one
import typing
import numpy as np
import scipy
import xarray as xr
import astra
import tqdm

from . import ut
from . import projections


def reconstruct_3d_cone(
        sinogram: xr.DataArray,
        projection_geometry,
        voxel_size: float,
        volume_size: typing.Iterable[float] = None,
        signal_is_differential: bool = False,
        gpu_index: int = 0,
        iterative = False,
        short_scan = False,
        n_iter=100,        
        padding: int = None, 
        own_hilbert: bool = False,
        padd_values = 'edge'):
    # the order of the arguments is (Y, X, Z)
    if volume_size is None:
        vol_geom = astra.create_vol_geom(
            sinogram.sizes["column"],
            sinogram.sizes["column"],
            sinogram.sizes["row"])
    else:
        vol_geom = astra.create_vol_geom(*volume_size)

    # astra expects the data to be in this order
    sinogram = sinogram.transpose("row", "angle", "column")

#     if signal_is_differential:
#         sinogram_data = np.imag(scipy.signal.hilbert(sinogram.values, axis=2)) / 2
#     else:
#         sinogram_data = sinogram.values
    
    if signal_is_differential:
        if padding is None:
            padding = sinogram.values.shape[1]
            
        if padding != 0:
            print('do padding')
            sinogram_data = np.pad(sinogram.values, ((0,0), (0,0), (padding, padding)), padd_values)
        else:
            print('no padding')
            sinogram_data = sinogram.values
            

        sinogram_data = np.imag(scipy.signal.hilbert(sinogram_data, axis=2)) / 2

        if padding != 0:
            sinogram_data = sinogram_data[:, :, padding:-padding]

    else:
        sinogram_data = sinogram.values
        

    


    sinogram_id = astra.data3d.create("-sino", projection_geometry, sinogram_data)
    reconstruction_id = astra.data3d.create('-vol', vol_geom)

    if iterative:
        algorithm = "SIRT3D_CUDA"
    else:
        if signal_is_differential:
            algorithm = "BP3D_CUDA"
        else:
            algorithm = "FDK_CUDA"

    cfg = astra.astra_dict(algorithm)

    cfg['ReconstructionDataId'] = reconstruction_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = {}
    
    if short_scan and algorithm == "FDK_CUDA":
        cfg['option']["ShortScan"] = short_scan
        
    cfg["option"]["GPUindex"] = gpu_index
    if iterative:
        cfg['option']['MinConstraint'] = 0
    algorithm_id = astra.algorithm.create(cfg)
    astra.algorithm.run(algorithm_id, n_iter if iterative else 1)
    reconstructed_data = astra.data3d.get(reconstruction_id)

    # change the sum to an integral with the physical interpretation
    if signal_is_differential:
        reconstructed_data /= sinogram.sizes["angle"]
    else:
        reconstructed_data /= voxel_size

    astra.data3d.delete(sinogram_id)
    astra.data3d.delete(reconstruction_id)
    astra.algorithm.delete(algorithm_id)

    return ut.volume_dataarray(reconstructed_data, voxel_size)
