# Copyright (c) 2025, ETH Zurich


from typing import Callable, Optional

import numpy as np
from numpy.ma.core import default_fill_value

import scipy
import scipy.constants as cnst
import scipy.interpolate as interp
import xarray as xr
import xraydb

from . import icru44data


class Material:
    def __init__(
        self,
        mu: Callable[[float], float],
        delta: Callable[[float], float],
        epsilon: Callable[[float, float, float, float, float, str, float], float],
    ) -> None:
        self._mu = mu
        self._delta = delta
        self._epsilon = epsilon

    def mu(self, E: float) -> float:
        return self._mu(E)

    def delta(self, E: float) -> float:
        return self._delta(E)

    def epsilon(
        self,
        E: float,
        d: Optional[float] = None,
        p: Optional[float] = None,
        D: Optional[float] = None,
        f: Optional[float] = None,
        mat_ref: Optional[str] = None,
        density_ref: Optional[float] = None,
    ) -> float:
        if d is not None:
            return self._epsilon(
                E, d, p, D, f, self._mu, self._delta, mat_ref, density_ref
            )
        else:
            return self._epsilon(E)
        
def xraydb_material(
    composition: str, density: float, epsilon_variant: Optional[str] = None
) -> Material:
    """
    Calculates delta and mu (1/m) using the xraydb and epsilon according to Lynch et al.
    """

    _mu = lambda E: xraydb.material_mu(composition, E / cnst.e) * 100
    _delta = lambda E: xraydb.xray_delta_beta(composition, density, E / cnst.e)[0]

    if epsilon_variant == None:
        _epsilon = lambda E: no_epsilon_available(E)

    elif epsilon_variant == "Lynch":
        _epsilon = (
            lambda E, d, p, D, f, _mu, _delta, mat_ref, density_ref: calc_epsilon_Lynch(
                E, d, p, D, f, _mu, _delta, mat_ref, density_ref
            )
        )

    else:
        raise ValueError(
            "Non-valid variant for epsilon provided. Options are None or Lynch."
        )

    return Material(_mu, _delta, _epsilon)

rho_e_water = 3.34e29


def HUp2rho_e(HUp):
    return rho_e_water * (HUp / 1000 + 1)


def rho_e2delta(rho_e, E):
    return (
        (
            cnst.physical_constants["classical electron radius"][0]
            * cnst.h**2
            * cnst.c**2
        )
        / (2 * np.pi * E**2)
        * rho_e
    )


def HUp2delta(HUp, E):
    return rho_e2delta(HUp2rho_e(HUp), E)


def delta2HUp(delta, E):
    return (delta / rho_e2delta(rho_e_water, E) - 1) * 1000


def mu2HU(mu, E):
    return (mu / water.mu(E) - 1) * 1000


# For delta values ref. Willner Phys. Med. Biol 59 (2014)
# 10.1088/0031-9155/59/7/1557
# they can be explicitly carried over to different energy
_Willner_HUp_adipose = -68.7
_Willner_HUp_fibroglandular = 65.2
_Willner_HUp_ductalcarcinoma_bulk = 47
_Willner_HUp_ductalcarcinoma_ducts = 72.5
_Willner_HUp_phyllodes_tumour = 44
_Willner_HUp_fibroadenoma_bulk = 45.1
_Willner_HUp_fibroadenoma_strands = 65
_Willner_HUp_lobularcarcinoma_tumour = 52
_Willner_HUp_lobularcarcinoma_fibroustissue = 68


# For attenuation values ref. P C Johns and M J Yaffe 1987 Phys. Med. Biol. 32 675
# They give values for 40 and 50keV, here I take the average between the two
# ref. 10.1088/0031-9155/32/6/002
_JohnsYaffe_energies_keV = np.r_[18, 20, 25, 30, 40, 50, 80, 110]
_JohnsYaffe_energies = _JohnsYaffe_energies_keV * 1e3 * cnst.e
_JohnsYaffe_mu_adipose_invcm = np.r_[
    0.558, 0.456, 0.322, 0.264, 0.215, 0.194, 0.167, 0.152
]
_JohnsYaffe_mu_adipose = _JohnsYaffe_mu_adipose_invcm * 100
_JohnsYaffe_mu_glandular_invcm = np.r_[
    1.028, 0.802, 0.506, 0.378, 0.273, 0.233, 0.189, 0.170
]
_JohnsYaffe_mu_glandular = _JohnsYaffe_mu_glandular_invcm * 100
_JohnsYaffe_mu_infiltratingDuctCarcinoma_invcm = np.r_[
    1.085, 0.844, 0.529, 0.392, 0.281, 0.238, 0.192, 0.173
]
_JohnsYaffe_mu_infiltratingDuctCarcinoma = (
    _JohnsYaffe_mu_infiltratingDuctCarcinoma_invcm * 100
)


adipose = Material(
    mu=interp.interp1d(_JohnsYaffe_energies, _JohnsYaffe_mu_adipose, kind = 'cubic', fill_value = 'extrapolate'),
    delta=lambda E: HUp2delta(_Willner_HUp_adipose, E),
    epsilon=lambda E: 0,
)

fibroglandular = Material(
    mu=interp.interp1d(_JohnsYaffe_energies, _JohnsYaffe_mu_glandular, kind = 'cubic', fill_value = 'extrapolate'),
    delta=lambda E: HUp2delta(_Willner_HUp_fibroglandular, E),
    epsilon=lambda E: 0,
)

ductalcarcinoma_bulk = Material(
    mu=interp.interp1d(_JohnsYaffe_energies, _JohnsYaffe_mu_infiltratingDuctCarcinoma, kind = 'cubic', fill_value = 'extrapolate'),
    delta=lambda E: HUp2delta(_Willner_HUp_ductalcarcinoma_bulk, E),
    epsilon=lambda E: 0,
)

ductalcarcinoma_ducts = Material(
    mu=interp.interp1d(_JohnsYaffe_energies, _JohnsYaffe_mu_infiltratingDuctCarcinoma,  kind = 'cubic', fill_value = 'extrapolate'),
    delta=lambda E: HUp2delta(_Willner_HUp_ductalcarcinoma_ducts, E),
    epsilon=lambda E: 0,
)

# calcifications=Material(
#     mu=interp.interp1d(_JohnsYaffe_energies, _JohnsYaffe_mu_infiltratingDuctCarcinoma),
#     delta=lambda E: HUp2delta(_Willner_HUp_ductalcarcinoma_ducts, E),
#     epsilon=lambda E: epsilon_GIBCT(R, sigma, ksi, E)
# )

phyllodes_tumour = Material(
    mu=interp.interp1d(_JohnsYaffe_energies, _JohnsYaffe_mu_infiltratingDuctCarcinoma,  kind = 'cubic', fill_value = 'extrapolate'),
    delta=lambda E: HUp2delta(_Willner_HUp_phyllodes_tumour, E),
    epsilon=None,
)

fibroadenoma_bulk = Material(
    mu=interp.interp1d(_JohnsYaffe_energies, _JohnsYaffe_mu_infiltratingDuctCarcinoma,  kind = 'cubic', fill_value = 'extrapolate'),
    delta=lambda E: HUp2delta(_Willner_HUp_fibroadenoma_bulk, E),
    epsilon=None,
)

fibroadenoma_strands = Material(
    mu=interp.interp1d(_JohnsYaffe_energies, _JohnsYaffe_mu_infiltratingDuctCarcinoma,  kind = 'cubic', fill_value = 'extrapolate'),
    delta=lambda E: HUp2delta(_Willner_HUp_fibroadenoma_strands, E),
    epsilon=None,
)

lobularcarcinoma_tumour = Material(
    mu=interp.interp1d(_JohnsYaffe_energies, _JohnsYaffe_mu_infiltratingDuctCarcinoma,  kind = 'cubic', fill_value = 'extrapolate'),
    delta=lambda E: HUp2delta(_Willner_HUp_lobularcarcinoma_tumour, E),
    epsilon=None,
)

lobularcarcinoma_fibroustissue = Material(
    mu=interp.interp1d(_JohnsYaffe_energies, _JohnsYaffe_mu_infiltratingDuctCarcinoma,  kind = 'cubic', fill_value = 'extrapolate'),
    delta=lambda E: HUp2delta(_Willner_HUp_lobularcarcinoma_fibroustissue, E),
    epsilon=lambda E: None,
)

nothing = Material(mu=lambda E: 0, delta=lambda E: 0, epsilon=lambda E: 0)

skin = fibroglandular