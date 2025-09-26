# Copyright (c) 2025, ETH Zurich

import glob
import os
import pathlib
import datetime

import numpy as np
import scipy
import scipy.constants as cnst
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage as ski
import xarray as xr
import astra
import gc
import sys

import utils.materials
import utils.phantoms
import utils.projections
import utils.phasestepping
import utils.optics
import utils.ut
import utils.reconstruction

import spekpy as sp

import skimage
from scipy import interpolate
import skimage.io as io
import xraydb
import utils.icru44data as icru44data
import argparse

from tqdm.auto import tqdm

def get_mu_deltas(compound, energies):
    
    assert compound in list(icru44data.icru44dict.keys()), "Compound not in ICRU 44 List"

    deltas = []
    mus = []
    _, _, density, composition = icru44data.icru44dict[compound]
    
    for eng in energies:
        photonenergy = eng * cnst.e
        wavelength = cnst.h * cnst.c / photonenergy
        
        deltas.append(
            np.sum([
                    weight * xraydb.xray_delta_beta(element, density, eng)[0]
                    for (element, weight)
                    in composition
            ])    
        )
        
        mus.append(
            np.sum([
                    weight * density * xraydb.mu_chantler(element, eng) * 100
                    for (element, weight)
                    in composition
            ]) 
        )
        
    deltas = np.asarray(deltas)
    mus = np.asarray(mus)
    
    return mus, deltas


def calculate_opening_angle(l, sample_size):
    op_angle = np.arcsin((sample_size /2) / l)
    return 2*op_angle

def _t(
        contrast: str,
        material: utils.materials.Material,
        m_background: utils.materials.Material,
        energy: float):

    if contrast == "mu":
        return np.abs(material.mu(energy) - m_background.mu(energy))
    elif contrast == "delta":
        return np.abs(material.delta(energy) - m_background.delta(energy))


def mu_au(eng):
    return xraydb.material_mu('Au', eng)*100

def mu_ai(eng):
    return xraydb.material_mu('Al', eng)*100

def mu_cu(eng):
    return xraydb.material_mu('Cu', eng)*100

def mu_h2o(eng):
    return xraydb.material_mu('H2O', eng)*100

def delta_h2o(eng):
    return xraydb.xray_delta_beta('H2O', 1, eng)[0]

def mu_breast(eng):
    mu_breast_tissue, delta_breast_tissue = get_mu_deltas('Breast Tissue (ICRU-44)', eng)
    return mu_breast_tissue

def mu_au(eng):
    return xraydb.material_mu('Au', eng)*100

def mu_si(eng):
    return xraydb.material_mu('Si', eng)*100

def mu_C(eng):
    return xraydb.material_mu('C', eng)*100


def second_derivative_3d_last_axis(arr, dx=1.0):
    """
    Compute the second derivative of a 3D array along the last axis using central differences.
    
    Parameters:
        arr (np.ndarray): 3D input array of shape (N1, N2, N3)
        dx (float): Spacing between points along the last axis
        
    Returns:
        np.ndarray: Array of the same shape with second derivative approximated
                    along the last axis. Boundary values are set to zero.
    """
    if arr.ndim != 3:
        raise ValueError("Input array must be 3-dimensional")
        
    d2f = np.zeros_like(arr)
    arr =  scipy.ndimage.gaussian_filter(arr, sigma=(0, 0, 0.47))
    
    print('Shape array is: ')
    print(arr.shape)
    
    # Central difference for interior points
    d2f[..., 1:-1] = (arr[..., :-2] - 2 * arr[..., 1:-1] + arr[..., 2:]) / dx**2
    
    # Optional: choose boundary treatment (here, zero second derivative)
    # Alternatives include one-sided differences or replicating interior values
    
    return d2f


# select the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def calc_Vis_theoretical(eng, Edes,m):
    V = 2/np.pi * np.abs(np.sin(np.pi / 2 * Edes / eng)**2 * np.sin(m * np.pi / 2 * Edes / eng))
    return V

def simulatesystem_polychromatic_multiple_slides_iterativ(
    sample_size_cm,
    source_sample_distance,
    sample_g2_distance,
    projected_px_size,
    nominal_flux,
    g2_pitch,
    spectrum,
    energies,
    E_des,
    To,
    padding,
    vis_spectrum = None,
    with_vis_hardening = False,
    detector_rows = 20,
    vis_penalty = 1.0,
    numberofprojections = 1000,
    visibility = 0.2,
    features_radius_mm = np.array([2, 1, 0.5, 0.25, 0.1]),
    fuse = False
):
    phantom_scaler = 1
    
    ### Calculate standard geometry parameteres
    source_detector_distance = source_sample_distance + sample_g2_distance
    print('Source detector distance: ', source_detector_distance)
    geometric_magnification = source_detector_distance / source_sample_distance
    sample_centre = source_detector_distance / geometric_magnification
    sample_size = sample_size_cm * 1e-2
    sample_radius = sample_size / 2
    projected_px_size = projected_px_size * 1e-6
    px_size = projected_px_size * geometric_magnification
    px_size_um = px_size * 1e6
    print('Px_size_um: ', px_size_um)

    detector_cols = int(np.ceil(sample_size * geometric_magnification / (px_size * 0.45 * 2)))
    detector_cols_phantom = int(np.ceil(sample_size * geometric_magnification / (px_size * 0.45 * 2 / phantom_scaler)))
    print('Detector cols: ', detector_cols)
    detector_size = detector_cols * px_size
    print('Detector size is: ', detector_size)
    
    ## I made a mistake in giving as input to SpekPy the distance in meters instead of centimeters. So the fluence will be higher than initially thought
    ## As everything is thought to be in much closer to the source. The fluence is still in photons / cm2. So to compensate for that error, the pixel size is given in 
    ## meters instead of cm, which accounts for the offset from meters to centimeters. So all good :). 
    intensity = nominal_flux * (0.1)**2 / source_detector_distance**2 * px_size**2
    spectrum = spectrum * (0.1)**2 / source_detector_distance**2 * px_size**2

    
    print('Px size is: ', px_size)
    
    ### Function for phantom creation
    ### Currently it covers 90% of the detector col number
    def myphantom(energy, contrast):
        return utils.phantoms.breast_sample(
            energy=energy,
            skin=utils.materials.skin,
            adipose=utils.materials.adipose,
            fibroglandular=utils.materials.fibroglandular,
            tumor=utils.materials.fibroadenoma_bulk,
            calcifications=utils.materials.fibroglandular,
            contrast=contrast,
            size=detector_cols_phantom,
            m_background=utils.materials.nothing,
            dtype=np.float32,
            eccentricity = 0.45/0.45,
            rows = int(detector_rows*phantom_scaler),
            voxel_size=projected_px_size/phantom_scaler,
            feature_sizes=features_radius_mm       
        )
    energy_e = energies * cnst.e
    angles = np.linspace(0, 2 * np.pi, numberofprojections).astype(np.float32)
    
    ### If no visibility spectrum is given, it will take the analytical visibility spectrum from Thomas Thüring
    ### https://doi.org/10.1098/rsta.2013.0027
    if vis_spectrum is None:
        vis_spectrum = calc_Vis_theoretical(energies, E_des, To)
    
    
    intensity = intensity / numberofprojections
    spectrum = spectrum / numberofprojections
    
    print('Intensity is: ', intensity)
    print('Spectrum intensity is: ', np.sum(spectrum) * (energies[1]-energies[0])/1e3)
    
    print('Sample size is: ', 2 * detector_cols * 0.45 * px_size / geometric_magnification)
    
    # Calculate the sample size to harden the spectrum for the calculation of delta
    #sample_size = 2 * detector_cols * 0.45 * px_size / geometric_magnification
    
    
    # Calculationg of the spectrum for mu and delta
    weights_mu = spectrum / np.sum(spectrum)
    weights_mu_ = weights_mu * np.exp(-mu_breast_tissue * sample_size)
    weights_delta = weights_mu_ / np.sum(weights_mu_) * vis_spectrum
    weights_delta = weights_delta / np.sum(weights_delta) #np.load('vis_spectrum_to_use_tmp.npy')
    
    #np.save('weights_mu_after_sample.npy', weights_mu_)
    #np.save('weights_delta_after_sample.npy', weights_delta)
    
    ### Calculate effective visibility
    if with_vis_hardening:
        visibility = np.sum(weights_mu * vis_spectrum) * vis_penalty
        visibility_after_sample = np.sum(weights_mu_ / np.sum(weights_mu_) * vis_spectrum) * vis_penalty
        sample_diffusion = visibility_after_sample / visibility
        print('Visibility: ', visibility)
        print('Visibility after sample: ', visibility_after_sample)
        
    else:
        visibility = visibility
        sample_diffusion = 1
    
    
    print('sample diffusion: ', sample_diffusion)

    att_lines = []
    delta_lines = []
    
    ### Iterate over all the energies and calculate the projection of delta and mu 
    ### -----------
    ### For delta we already weight it by the normalized spectrum * vis_spectrum since it would be an extremely 
    ### lengthy calculation, since for every projection and pixel we would need to a priori know the hardened visibility spectrum
    ### for the signal retrieval
    ### -----------
    ### For attenuation we collect all the projetions for all energies, and later we weight it by the spectrum 
    ### for all energies
    projection_geometry = utils.projections.cone_projection_geometry(
        detector_cols=detector_cols,
        detector_rows=detector_rows,
        angles=angles,
        detector_px_size=px_size,
        voxel_size=projected_px_size,
        distance_source_rotcentre=source_sample_distance,
        distance_rotcentre_detector=sample_g2_distance
    )
    
    
    
    for i, (eng, w_m, w_d) in enumerate(zip(energy_e, weights_mu, weights_delta)):
        if i > 0:
            del phantom
            gc.collect()
        phantom = utils.ut.volume_dataset(
            dict(attenuation=myphantom(eng, "mu"), delta=myphantom(eng, "delta")),
            voxel_size=px_size / geometric_magnification / phantom_scaler,
        )
        phantom.attenuation.attrs["units"] = "1/m"
        phantom.delta.attrs["units"] = "1"
    
        #phantom.delta.values = scipy.ndimage.gaussian_filter(phantom.delta.values, sigma = 0.6)

        att_coeff_line_integrals = utils.projections.project_cone3d(
            vdata=phantom.attenuation,
            detector_px_size=px_size,
            detector_cols=detector_cols,
            detector_rows = detector_rows,
            angles=angles,
            use_cuda=True,
            source_origin = source_sample_distance,
            origin_detector = sample_g2_distance,
            voxel_size = px_size / geometric_magnification / phantom_scaler,
            name="att. coeff. line integral",
        )

        delta_line_integrals = utils.projections.project_cone3d(
            vdata=phantom.delta,
            detector_px_size=px_size,
            detector_cols=detector_cols,
            detector_rows = detector_rows,
            angles=angles,
            use_cuda=True,
            source_origin = source_sample_distance,
            origin_detector = sample_g2_distance,
            voxel_size = px_size / geometric_magnification / phantom_scaler,
            name="delta line integral",
        )

        
        delta_lines.append(delta_line_integrals)
        
        line_integrals = xr.Dataset(
            dict(attenuation=att_coeff_line_integrals, delta=delta_line_integrals)
        )
        
        attenuation_projections = utils.projections.mu_line_integrals_to_attenuation(
            line_integrals.attenuation
        )
        att_lines.append(attenuation_projections)
        
        


    
    ##### Calculate spectrum per pixel for visibility hardening
    spec_per_pixel = [att * weights_mu[k] for k, att in enumerate(att_lines)]
    spec_per_pixel = np.asarray(spec_per_pixel)
    spec_per_pixel = spec_per_pixel / np.sum(spec_per_pixel, axis = 0)
    #np.save('spec_per_pixel.npy', spec_per_pixel.astype(np.float32))
    spec_per_pixel = spec_per_pixel * vis_spectrum[:, np.newaxis, np.newaxis, np.newaxis]
    sample_diffusion = np.sum(spec_per_pixel, axis = 0) / visibility * vis_penalty
    spec_per_pixel = spec_per_pixel / np.sum(spec_per_pixel, axis = 0)
    #np.save('spec_per_pixel_vis.npy', spec_per_pixel.astype(np.float32))
    #np.save('sample_diffusion.npy', sample_diffusion.astype(np.float32))
    
    print('spec per pixel has size: ', spec_per_pixel.shape)
    print('sampel diffusion has size: ', sample_diffusion.shape)
    
    delta_lines = xr.concat(delta_lines, dim='energies')
    
    #io.imsave('delta_integrals.tif', delta_lines.values.astype(np.float32))
    
    #delta_line_integrals = delta_lines.sum(dim='energies')
    
    refraction_angle = delta_lines.differentiate("column") * geometric_magnification
    refraction_angle.name="refraction_angle"
    refraction_angle.attrs["units"] = "rad"
    
    pattern_phase_shift = utils.optics.refraction_angle_to_pattern_phase_shift(
        refraction_angle=refraction_angle,
        sample_analyser_distance=sample_g2_distance,
        pattern_pitch=g2_pitch,
    )
    
    print('Calculating temporary phase shift from complex values')
    tmp_phase_shift = np.angle(np.sum(spec_per_pixel * np.exp(1j * pattern_phase_shift.values), axis = 0))
    print('tmp phase shift has size' )
    print(tmp_phase_shift.shape)
    del refraction_angle
    

    #tmp_weighted_lines = np.sum(refraction_angle.values * spec_per_pixel, axis = 0)
    pattern_phase_shift = xr.DataArray(
        tmp_phase_shift,
        coords=delta_line_integrals.coords,
        dims=delta_line_integrals.dims,
        attrs=delta_line_integrals.attrs,
    )
    
    print('Pattern phase shift has the following shape: ')
    print(pattern_phase_shift.shape)
    
    
    del delta_lines
    del tmp_phase_shift
    del spec_per_pixel
    
    #del concatenated_xarray
    gc.collect()
    
    
    del delta_line_integrals
    gc.collect()

    
    # Add a blurring to account for the loss in resolution in phase
    filtered_phase = scipy.ndimage.gaussian_filter(pattern_phase_shift.values, sigma=(0, 0, 0.47))

    # Rewrap into xarray with same coords and dims
    pattern_phase_shift = xr.DataArray(
        filtered_phase,
        coords=pattern_phase_shift.coords,
        dims=pattern_phase_shift.dims,
        attrs=pattern_phase_shift.attrs,
    )
    
    
    del filtered_phase
    # Calculate second derivate for visibility reduction
    second_dev_phase = pattern_phase_shift.differentiate("column") * geometric_magnification / (2*np.pi) #refraction_angle.differentiate("column") * geometric_magnification

    vis_red_tmp = np.abs(np.sinc(np.abs(second_dev_phase * projected_px_size)))
    second_dev_phase = xr.DataArray(
        vis_red_tmp,
        coords=pattern_phase_shift.coords,
        dims=pattern_phase_shift.dims,
        attrs=pattern_phase_shift.attrs
    )
    
    del vis_red_tmp
    
    
    #io.imsave('second_derivative.tif', second_dev_phase.values.astype(np.float32))
    
    ### Calculate number of counts for all energies and projetion of attenuation. This gives a beam hardened 
    ### intensity profile.
    
    patterned_att = []
    for k, att_proj in enumerate(att_lines):
        
        patterned_wavefront = utils.optics.flat_patterned_wavefront(
            intensity=spectrum[k]*(energies[1]-energies[0])/1e3,
            intensity_unit="counts",
            phase=0,
            visibility=visibility,
            cols=detector_cols,
            rows=detector_rows
        )
        
        patterned_wavefront_after_sample = (
            utils.optics.propagate_patterned_wavefront_through_sample(
                patterned_wavefront=patterned_wavefront,
                sample_attenuation=att_proj,
                sample_pattern_shift=pattern_phase_shift,
                sample_diffusion=sample_diffusion * second_dev_phase,
            )
        )
        
        patterned_att.append(patterned_wavefront_after_sample.intensity)
    
    del pattern_phase_shift
    del att_lines
    gc.collect()
        
    patterned_att = xr.concat(patterned_att, dim = 'energies')
    patterned_att = patterned_att.sum(dim='energies')
    
    patterned_wavefront_after_sample['intensity'] = patterned_att
    patterned_wavefront_after_sample['intensity'].attrs["units"] = "counts"
    
    del patterned_att
    gc.collect()
    
    

    patterned_wavefront = utils.optics.flat_patterned_wavefront(
            intensity=np.sum(spectrum)*(energies[1]-energies[0])/1e3,
            intensity_unit="counts",
            phase=0,
            visibility=visibility,
            cols=detector_cols,
            rows = detector_rows
    )
    
    print('Spectrum total: ', np.sum(spectrum)*(energies[1]-energies[0])/1e3)
    
    shape_wavefront = patterned_wavefront_after_sample.intensity.shape[2]
    
    print('Shape wavefront: ', patterned_wavefront_after_sample.intensity.shape)
    
    #print('Int after sample: ', np.mean(patterned_wavefront_after_sample.intensity[int(detector_rows/2), :, int(shape_wavefront/2)-10:int(shape_wavefront/2)+10], axis = (0,1)))
    print('Int after sample: ', np.mean(patterned_wavefront_after_sample.intensity[int(detector_rows/2), 0, int(shape_wavefront/2)-10:int(shape_wavefront/2)+10]))

    # simulate phase-stepping
    phase_steps = utils.phasestepping.equidistant_phase_stepping_array(3)

    phase_step_intensity = utils.phasestepping.phase_stepping_model_by_mean(
        phase_steps=phase_steps, wavefront=patterned_wavefront_after_sample
    )
    
    
    mean_values = [np.mean(phase_step_intensity[int(detector_rows/2), 0, int(shape_wavefront/2)-10:int(shape_wavefront/2)+10, i]) for i in range(3)]
    print('mean of all values: ', np.mean(mean_values))
    print('sum of all values ', np.sum(mean_values))
    
    # add noise
    phase_step_noisy_intensity = utils.ut.add_poisson_noise(phase_step_intensity)

    # retrieve the signal
    retrieved_signal = utils.phasestepping.signal_retrieval_da_by_mean(
        phase_step_noisy_intensity
    )
    
    del phase_step_noisy_intensity

    # remove exact zeros, as they produce artefacts
    retrieved_signal["intensity"] = retrieved_signal.intensity.where(
        retrieved_signal.intensity > 0
    ).fillna(1)
    

    retrieved_signal["visibility"] = retrieved_signal.visibility.where(
        retrieved_signal.visibility > 0
    ).fillna(1e-3)

    
    #io.imsave('retrieved_visibility_values.tif', retrieved_signal.visibility.values.astype(np.float32))
    print('retrieved intensity: ', np.mean(retrieved_signal.intensity[int(detector_rows/2), 0, int(shape_wavefront/2)-10:int(shape_wavefront/2)+10]))

    relative_change_in_wavefront = utils.optics.relate_to_reference_patterned_wavefront(
        reference_patterned_wavefront=patterned_wavefront,
        patterned_wavefront_after_sample=retrieved_signal,
    )
    
    #io.imsave('retrieved_diffusion_values.tif', relative_change_in_wavefront.diffusion.values.astype(np.float32))

    relative_change_in_wavefront[
        "refraction_angle"
        ] = utils.optics.pattern_phase_shift_to_refraction_angle(
            relative_change_in_wavefront.pattern_phase_shift,
            sample_analyser_distance=sample_g2_distance,
            pattern_pitch=g2_pitch,
        )
    
    rows_coord = relative_change_in_wavefront.coords['row'].values
    attenuation_reconstruction = []
    delta_reconstruction = []
    
    del retrieved_signal
    del patterned_wavefront
    gc.collect()
    
    #io.imsave('refraction_angle_sino_cone_nonoise.tif', relative_change_in_wavefront.refraction_angle.values)
    print('refration dtype: ', relative_change_in_wavefront.refraction_angle.dtype)
    
    relative_change_in_wavefront.refraction_angle.values = relative_change_in_wavefront.refraction_angle.astype(np.float32)
    print('refration dtype: ', relative_change_in_wavefront.refraction_angle.dtype)
    
    attenuation_reconstruction = utils.reconstruction.reconstruct_3d_cone(
        sinogram=relative_change_in_wavefront.attenuation.astype(np.float32),
        projection_geometry=projection_geometry,
        voxel_size=projected_px_size,
        signal_is_differential=False,
        iterative=False)

    delta_reconstruction = utils.reconstruction.reconstruct_3d_cone(
        sinogram=relative_change_in_wavefront.refraction_angle,
        projection_geometry=projection_geometry,
        voxel_size=projected_px_size,
        signal_is_differential=True,
        iterative=False,
        own_hilbert=False,
        padding = detector_cols)
    
    dark_field_reconstruction = utils.reconstruction.reconstruct_3d_cone(
        sinogram=relative_change_in_wavefront.diffusion.astype(np.float32),
        projection_geometry=projection_geometry,
        voxel_size=projected_px_size,
        signal_is_differential=False,
        iterative=False)

    reconstruction = xr.Dataset(dict(
        attenuation=attenuation_reconstruction,
        delta=delta_reconstruction,
        df=dark_field_reconstruction
    ))
    
    print('Delta reconstruction is: ', delta_reconstruction.dtype)
    
    del relative_change_in_wavefront
    del attenuation_reconstruction
    del delta_reconstruction
    del dark_field_reconstruction
    gc.collect()
    
    if fuse:
        optsigma = 5
        da_low = scipy.ndimage.gaussian_filter(
            reconstruction.attenuation.data, sigma=(0,optsigma,optsigma)
        )

        #dp_axial_filtered = scipy.ndimage.gaussian_filter(reconstruction.delta.data, sigma=(1,0,0))
        dp_low = scipy.ndimage.gaussian_filter(reconstruction.delta.data, sigma=(0,optsigma,optsigma))
        dp_high = reconstruction.delta - dp_low
        #dp_high = dp_axial_filtered - dp_low
        fused = reconstruction.attenuation.copy()

        weighted_mu = np.sum(utils.materials.fibroglandular.μ(energy_e) * weights_mu)
        weighted_delta = np.sum(utils.materials.fibroglandular.δ(energy_e) * weights_delta)

        fused.data = da_low + 2 * dp_high * weighted_mu / weighted_delta

        del da_low
        del dp_high

        return reconstruction.attenuation, fused, reconstruction.delta, reconstruction.df, phantom
    else:
        return reconstruction.attenuation, None, reconstruction.delta, reconstruction.df, phantom



############################################
#### HERE RUN THE CODE ####################
############################################

if __name__ == "__main__":
    
    current_path = os.getcwd()
    
    parser = argparse.ArgumentParser(description="Row numbers for sinogram processing")
    parser.add_argument("--measurementname", required=True, type=str, help = "System name")
    parser.add_argument("--samplesize", required=True, type=int, help = "Sample size in cm")
    parser.add_argument("--doseratio", required=True, type=float, help = "Dose ratio to Rawlik")
    parser.add_argument("--vp", required=True, type=float, help = "Visibility penalty")
    parser.add_argument("--doses", required=True, type=str, help="Comma-separated list of float values")
    
    #GI Parameters
    parser.add_argument("--gratingheight", required = True, type=float, help = "Grating height")
    parser.add_argument("--ss", required = True, type=float, help = "Source Sample Distance")
    parser.add_argument("--sd", required = True, type=float, help = "Sample G2 Distance")
    parser.add_argument("--pitch", required = True, type=float, help = "G2 Pitch")
    parser.add_argument("--to", required = True, type=float, help = "Talbot order")
    parser.add_argument("--ed", required = True, type=int, help = "Design Energy")
    parser.add_argument("--kvp", required = True, type=int, help = "kVp")
    parser.add_argument("--vis_spectrum", required = False, type=str, default = 'utils/vis_spectrum/vis_spectrum_gibct_v2.npy', help = "Visibility spectrum path",)
    parser.add_argument("--energy_spectrum", required = False, type=str, default = 'utils/vis_spectrum/energies_spectrum_gibct_v2.npy', help = "Energy spectrum path")

    parser.add_argument("--pixel", required = True, type=int, help = "Pixel size in um")
    parser.add_argument("--vishardening", action="store_true", help="If visibility spectrum is used")
    parser.add_argument("--visibility", required = False, type=float, default=0.2, help="Flat visibility value")
    parser.add_argument("--nogratings", action="store_true", help="Use when no gratings shall be put in the beam")

    args = parser.parse_args()

    # Output all parameters
    print("Measurement name: ", args.measurementname)
    print("Sample size: ", args.samplesize)
    print("Dose ratio: ", args.doseratio)
    print("Visibility penalty: ", args.vp)
    print("Doses: ", args.doses)
    print("Grating height: ", args.gratingheight)
    print("Source Sample Distance: ", args.ss)
    print("Sample G2 Distance: ", args.sd)
    print("G2 Pitch: ", args.pitch)
    print("Talbot order: ", args.to)
    print("Design Energy: ", args.ed)
    print("kVp: ", args.kvp)
    print("Pixel size: ", args.pixel)
    print("Vis spectrum path ", args.vis_spectrum)
    print("Energy spectrum for vis path: ", args.energy_spectrum)
    print("Visibility hardening: ", args.vishardening)
    print('Inpuuted viisibility: ', args.visibility)
    print('No gratings?: ', args.nogratings)

    measurement_name = args.measurementname
    sample_size_cm = args.samplesize
    dose_ratio_to_rawlik = args.doseratio
    visibility_penalty = args.vp
    doses = [float(x) for x in args.doses.split(",")]
    
    print(f"Sample size is: {sample_size_cm}")
    
    
    storage = "<PATH WHERE TO SAVE FILES>"

    # System parameters
    height = args.gratingheight #140e-6
    source_sample_distance = args.ss #0.64567 + 0.15 + 0.1
    sample_g2_distance = args.sd #0.64567 - 0.1
    M = (source_sample_distance + sample_g2_distance) / source_sample_distance
    g2_pitch = args.pitch
    To = args.to
    E_des = args.ed
    top_kvp = args.kvp
    
    pixel_size = args.pixel

    with_vis_hardening = args.vishardening
    visibility = args.visibility

    projected_px_size_um = pixel_size / M
    print('projected_px_size: ', projected_px_size_um)

    # We reference to the Static Setup of Rawlik et al. 
    # mAs there is 50*6*10 for 22mGy @ 70kVp
    # We take then the calculated ratios from the previous notebook
    total_time = 5*60
    current = 10
    mas = current * total_time / 2.2 / dose_ratio_to_rawlik # So we are at 10mGy for Rawlik et al.



    s = sp.Spek(kvp=top_kvp, dk = 0.1, th = 10, z = 0.1, mas = mas)
    s.filter('Al', 3) # Create a spectrum
    k, f = s.get_spectrum(edges=True, diff = True, flu = True) # Get the spectrum
    tube_spectrum_txt = interpolate.interp1d(k*1e3, f, fill_value = 'extrapolate')
    energies = np.arange(15, top_kvp+1, 1)*1e3
    att_grating = (np.exp(-mu_au(energies)*height) + np.exp(-mu_si(energies) * height))/2 * np.exp(-mu_si(energies) * 200e-6)

    mu_breast_tissue, delta_breast_tissue = get_mu_deltas('Breast Tissue (ICRU-44)', energies)
    if args.nogratings:
        print('Used no gratings')
        spectrum = tube_spectrum_txt(energies)
    else:
        spectrum = tube_spectrum_txt(energies) * att_grating**2 #
    
    spectrum = spectrum.astype(np.float32)
    nominal_flux = np.sum(spectrum) * (energies[1]-energies[0])/1e3
    
    if args.vis_spectrum == "None":
        vis_spectrum = None
    else:
        vis_spectrum_txt = np.load(args.vis_spectrum)
        energies_vis = np.load(args.energy_spectrum)
        vis_spectrum_txt = interpolate.interp1d(energies_vis, vis_spectrum_txt, fill_value = 'extrapolate')
        vis_spectrum = vis_spectrum_txt(energies).astype(np.float32)
        

    detector_penalty = 1.0
    features_radius_mm = np.array([3, 2, 1, 0.5, 0.25, 0.1])

    att_large = []
    ref_large = []
    fus_large = []

    dose = doses[0]
    
    att, fus, ref, df, phantom = simulatesystem_polychromatic_multiple_slides_iterativ(
        sample_size_cm = sample_size_cm,
        source_sample_distance=source_sample_distance,
        sample_g2_distance=sample_g2_distance,
        projected_px_size=projected_px_size_um,
        nominal_flux=nominal_flux * dose,
        g2_pitch=g2_pitch,
        spectrum=spectrum * dose,
        energies=energies,
        E_des=E_des,
        To=To,
        padding = None,
        vis_spectrum = vis_spectrum,
        with_vis_hardening = with_vis_hardening,
        detector_rows = 41,
        vis_penalty = visibility_penalty,
        visibility=visibility,
        features_radius_mm = features_radius_mm
    )
    
    


    det_cols = phantom.attenuation.shape[1]
    voxSize = projected_px_size_um * 1e-6

    idx_row = 4

    os.makedirs(os.path.join(storage, f"{measurement_name}_tumor"), exist_ok=True)

    io.imsave(os.path.join(storage, f"{measurement_name}_tumor/{measurement_name}_{sample_size_cm}_phantom.tif"), phantom.attenuation[idx_row,:,:])

    io.imsave(os.path.join(storage, f"{measurement_name}_tumor/{measurement_name}_tumor_{sample_size_cm}_absorption_{int(10*doses[0])}mGy.tif"), att)
    
    if not args.nogratings:
        io.imsave(os.path.join(storage, f"{measurement_name}_tumor/{measurement_name}_tumor_{sample_size_cm}_refraction_{int(10*doses[0])}mGy.tif"), ref)
        io.imsave(os.path.join(storage, f"{measurement_name}_tumor/{measurement_name}_tumor_{sample_size_cm}_diffusion_{int(10*doses[0])}mGy.tif"), df)
        

    del att, fus, ref, phantom, df
    gc.collect()

    print(f"Done with dose: {dose}")