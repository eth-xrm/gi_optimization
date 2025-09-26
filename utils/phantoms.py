# Copyright (c) 2025, ETH Zurich


import os
import typing
from typing import Tuple, Optional, Union

import numpy as np
import numpy.typing as npt

import skimage.draw

from . import materials


def _t(
        contrast: str,
        material: materials.Material,
        m_background: materials.Material,
        energy: float):

    if contrast == "mu":
        return material.mu(energy) - m_background.mu(energy)
    elif contrast == "delta":
        return material.delta(energy) - m_background.delta(energy)

    
def breast_sample(
    energy: float,
    skin: materials.Material,
    adipose: materials.Material,
    fibroglandular: materials.Material,
    tumor: materials.Material,
    calcifications: materials.Material,
    contrast: str,
    size: int,
    m_background: materials.Material=materials.nothing,
    dtype=np.float32,
    eccentricity: float = 0.35 / 0.45,
    rows: int = 1,
    voxel_size: float = 200e-6,
    feature_sizes: np.ndarray = np.array([3, 1.5, 0.75, 0.5, 0.25])
    ) -> np.ndarray:
    
    def t(material):
        if material in [materials.fibroadenoma_bulk, materials.ductalcarcinoma_bulk] and contrast == "delta":
            t_tumor = _t(contrast, material, m_background, energy)
            t_fibroglandular = _t(contrast, materials.fibroglandular, m_background, energy)
            t_diff = t_fibroglandular - t_tumor
            return t_fibroglandular + t_diff
        else:
            return _t(contrast, material, m_background, energy)

    phantom = np.zeros((size, size), dtype=np.float32)
    # skin
    phantom[skimage.draw.ellipse(
        size / 2, size / 2,
        size * 0.45, size * 0.45 * eccentricity)] = t(skin)

    # fat
    phantom[skimage.draw.ellipse(
        size / 2, size / 2,
        size * 0.45 * 0.95, size * 0.45 * eccentricity * 0.95)] = t(adipose)

    # read phantom channels image if an array image is not given
    phantom_image = None
    if phantom_image is None:

        phantom_image_filename = os.path.join(
            os.path.dirname(__file__), "images/phantom_breast.npy"
        )
        phantom_image = np.load(phantom_image_filename)

    resized_img = skimage.transform.resize(phantom_image, (size, size), order=0, preserve_range=True, anti_aliasing=False)
    resized_img[phantom != t(adipose)] = 0


    phantom[resized_img == 1] = t(fibroglandular)
    #phantom[resized_img == 2] = t(calcifications)

    #features_radius_mm = np.array([2, 1, 0.5, 0.25, 0.1])

    # Those are the small features that we try to keep the same size
    features_px = (feature_sizes* 1e-3) / voxel_size

    

    # Inside fibrogland
    phantom[skimage.draw.disk(
        (size / 2 + 2 * features_px[0], size / 2 + 0.18 * size),
        features_px[0])] = t(tumor)

    phantom[skimage.draw.disk(
        (size / 2, size / 2 + 0.18 * size),
        features_px[1])] = t(tumor)

    phantom[skimage.draw.disk(
        (size / 2 - 1.2 * features_px[0], size / 2 + 0.18 * size),
        features_px[2])] = t(tumor)

    phantom[skimage.draw.disk(
        (size / 2 - 2 * features_px[0], size / 2 + 0.18 * size),
        features_px[3])] = t(tumor)

    phantom[skimage.draw.disk(
        (size / 2 - 2.5*features_px[0], size / 2 + 0.18 * size),
        features_px[4])] = t(tumor)

    phantom[skimage.draw.disk(
        (size / 2 - 3 *features_px[0], size / 2 + 0.18 * size),
        features_px[5])] = t(tumor)


    # Outside fibrogland
    phantom[skimage.draw.disk(
        (size / 2 + 3.5 * features_px[0], size / 2 - 0.22 * size),
        features_px[0])] = t(tumor)

    phantom[skimage.draw.disk(
        (size / 2 + 1.5 * features_px[0], size / 2 - 0.22 * size),
        features_px[1])] = t(tumor)

    phantom[skimage.draw.disk(
        (size / 2, size / 2 - 0.22 * size),
        features_px[2])] = t(tumor)

    phantom[skimage.draw.disk(
        (size / 2 - 1 * features_px[0], size / 2 - 0.22 * size),
        features_px[3])] = t(tumor)

    phantom[skimage.draw.disk(
        (size / 2 - 1.5*features_px[0], size / 2 - 0.22 * size),
        features_px[4])] = t(tumor)

    phantom[skimage.draw.disk(
        (size / 2 - 2 *features_px[0], size / 2 - 0.22 * size),
        features_px[5])] = t(tumor)


    # Large tumor in an area
    phantom[skimage.draw.disk(
        (size / 2 - 0.04*size, size / 2 + 0.04 * size),
         size * 0.14 / 2)] = t(tumor)


    if rows > 1:
        phantom_tmp = [phantom for i in range(rows)]
        phantom = np.asarray(phantom_tmp)
        del phantom_tmp

    return phantom