# Copyright (c) 2025, ETH Zurich


# Here we have the forward model for projecting volumes onto a detector
import typing
import numpy as np
import scipy.spatial.transform as transform
import xarray as xr
import astra

def delta_line_integral_to_refraction_angle(
        delta_integrals: xr.DataArray,
        px_size: float = None,
        name: str = "refraction angle",
        unit: str = "rad") -> xr.DataArray:
    if px_size is None:
        px_size = np.median(np.diff(delta_integrals.coords["column"]))

    refraction_angle = delta_integrals.differentiate("column")

    refraction_angle.name=name
    refraction_angle.attrs["units"] = unit

    return refraction_angle


def mu_line_integrals_to_attenuation(
        line_integrals: xr.DataArray,
        name: str = "attenuation",
        unit: str = "1") -> xr.DataArray:
    attenuation = np.exp(-line_integrals)

    attenuation.name = name
    attenuation.attrs["units"] = unit

    return attenuation

def cone_projection_geometry(
        detector_cols: int,
        detector_rows: int,
        angles: typing.Iterable[float],
        detector_px_size: float,
        voxel_size: float,
        distance_source_rotcentre: float,
        distance_rotcentre_detector: float,
        detector_vertical_offset: float = 0,
        detector_horizontal_offset: float = 0,
        detector_roll_angle: float = 0,
        detector_tilt_angle: float = 0):
    # The coordinate system is centred at the centre of the reconstruction volume.
    # We let the rotation axis define the vertical (z here) direction.
    # transformations for every frame
    system_rot_axis = np.r_[0, 0, 1]
    system_transform = [ transform.Rotation.from_rotvec(a * system_rot_axis) for a in np.array(angles) ]
    
    # the position of the source
    # The reconstruction volume is centred around (0,0,0),
    # so when when we create an offset we move both the source and the centre of the
    # detector such that the line connecting those points goes through (0,0,0)
    src = np.r_[
        0,
        -distance_source_rotcentre,
        -detector_vertical_offset / distance_rotcentre_detector * distance_source_rotcentre
    ]
    src = np.stack([ t.apply(src) for t in system_transform ])
    
    # the position of the detector
    det = np.r_[
        detector_horizontal_offset,
        distance_rotcentre_detector,
        detector_vertical_offset
    ]
    det = np.stack([ t.apply(det) for t in system_transform ])
    
    # det_h is a vector from detector pixel (0,0) to (0,1)
    # det_v is a vector from detector pixel (0,0) to (1,0)
    if np.isscalar(detector_px_size):
        det_v = np.r_[0, 0, detector_px_size]
        det_h = np.r_[detector_px_size, 0, 0]
    else:
        det_v = np.r_[0, 0, detector_px_size[0]]
        det_h = np.r_[detector_px_size[1], 0, 0]
    
    # apply detector roll
    det_roll = transform.Rotation.from_rotvec(detector_roll_angle * np.r_[0, 1, 0])
    det_h = det_roll.apply(det_h)
    det_v = det_roll.apply(det_v)

    det_tilt = transform.Rotation.from_rotvec(detector_tilt_angle * np.r_[1, 0, 0])
    det_h = det_tilt.apply(det_h)
    det_v = det_tilt.apply(det_v)
    
    det_h = np.stack([ t.apply(det_h) for t in system_transform ])
    det_v = np.stack([ t.apply(det_v) for t in system_transform ])
    
    # Construct the vectors object according to Astra's definition
    vectors = np.hstack([src, det, det_h, det_v])
    
    # Change the unit to the voxel size, as per Astra's definition
    vectors /= voxel_size

    proj_geom = astra.create_proj_geom("cone_vec", detector_rows, detector_cols, vectors)

    return proj_geom

def project_cone3d(
        vdata: xr.DataArray,
        angles: typing.Iterable[float],
        detector_px_size: float,
        detector_cols: int,
        detector_rows: int,
        source_origin: float,
        origin_detector: float,
        voxel_size: float = None,
        use_cuda: bool = False,
        name: str = None,
        unit: str = None,
        ) -> xr.DataArray:

    if voxel_size is None:
        voxel_size = np.median(np.diff(vdata.coords["x"]))
    
    #print(voxel_size)
    
    
    vx = vdata.shape[-2]
    vy = vdata.shape[-1]
    vz = vdata.shape[-3]
    
    min_x = -vx / 2.0 * voxel_size# divide by 2 since are going left and right -> nothing to do with binning
    max_x = vx / 2.0 * voxel_size
    min_y = -vy / 2.0 * voxel_size
    max_y = vy / 2.0 * voxel_size
    min_z = -vz / 2.0 * voxel_size
    max_z = vz / 2.0 * voxel_size
    
    #print(vdata.shape[-2], vdata.shape[-1], vdata.shape[-3])
    vol_geom = astra.create_vol_geom(vy, vx, vz, min_y, max_y, min_x, max_x, min_z, max_z)
    #vdetector_px_size_in_astra_units = voxel_size / detector_px_size
    
    #vdetector_px_size_in_astra_units =  detector_px_size / voxel_size

    proj_geom = astra.create_proj_geom(
            "cone",
            detector_px_size,
            detector_px_size,
            detector_rows,
            detector_cols,
            angles,
            source_origin, # detector_px_size,
            origin_detector # detector_px_size,
            )
        
    #proj_id = astra.create_projector(
    #    "cuda" if use_cuda else "line", proj_geom, vol_geom)

    phantom_id = astra.data3d.create('-vol', vol_geom, data=vdata.values)
    sino_id, projection_data = astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)
    #sino_id, projection_data = astra.creators.create_sino(phantom_id, proj_id)

    # change the sum to an integral with the physical interpretation
    #projection_data *= voxel_size

    projection_da = xr.DataArray(
        data=projection_data.astype(np.float32),
        dims=("row", "angle", "column"),
        coords=dict(
            row = np.arange(detector_rows, dtype=np.float32) * detector_px_size,
            angle=angles,
            column=np.arange(detector_cols, dtype = np.float32) * detector_px_size),
        name=name,
        attrs=dict(units=unit))

    projection_da.coords["column"].attrs["units"] = "m"
    projection_da.coords["row"].attrs["units"] = "m"
    

    return projection_da
