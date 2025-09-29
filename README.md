
This repository contains code and resources for the paper:

> Jia-Xin Guo et al. (2025). "A Promising Downscaling Strategy for Topographic Heavy Rainfalls over the Asian-Australian Monsoon Region by Leveraging Mutli-Scale Moisture Dynamical Control" *(Submitted)*.

## Structure
- `data/` raw and processed data, please refer to `notebooks` for detailed usage
  - Two TaiESM1 IVT files are archived at doi:10.5281/zenodo.17199183 due to file sizes.
- `src/` source code (data preprocessing, VAE model related, plotting utility functions)
- `notebooks/` reproducible analysis and figures
- `environments/` conda environments for different machines

## Important Notice
There are three environment files: env_main.yml, env_vae.yml, env_taiwanvvm.yml, which are meant to execute different code files, corresponding to the directory names.
For example: The environment `main` should be used to execute code in `src/main/` and `notebooks/main/`.
