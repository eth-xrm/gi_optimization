<!-- Copyright (c) 2025, ETH Zurich -->

# Simultaneous signal optimization of refraction and attenuation in X-ray grating interferometry: a case study for breast imaging

This repository contains the reconstruction and analysis code for the manuscript 
*Simultaneous signal optimization of refraction and attenuation in X-ray grating interferometry: a case study for breast imaging*.

For the simulation of the visibility spectrum please refer to RAVE SIM package (https://doi.org/10.1364/OE.543500).

The Python environment is shared across the entire codebase. The environment.yml contains all necessary packages.

```bash
conda env create -f environment.yml
```

All notebooks need to be adapted to incorporate the correct paths to the folders.

The bash-scripts for the reconstuction are implemented to be used in a Cluster managed by SLURM. If no cluser available, the python scripts can be run individually.