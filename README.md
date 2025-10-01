# Multiscale-MoistDyn-Control

This repository contains code and resources for the paper:

> Guo et al. (2025). "A Promising Downscaling Strategy for Topographic Heavy Rainfalls over the Asian-Australian Monsoon Region by Leveraging Mutli-Scale Moisture Dynamical Control" *(Submitted)*.

## Structure
- `environments/`  
  - `env_main.yml`      (used for code in `notebooks/main` and `src/main`)  
  - `env_taiwanvvm.yml` (used for code in `notebooks/taiwanvvm` and `src/taiwanvvm`)  
  - `env_vae.yml`       (used for code in `src/vae`)  
- `data/`  
  - `raw/`              (unprocessed data and public URLs)  
  - `processed/`        (ready-to-use and demo. data)  
  - `vae_model/`        (model.h5, training log, dataset [train/val/test])  
- `src/`                (source code for processing data, model training, and plotting utilities)  
  - `main/`  
  - `taiwanvvm/`  
  - `vae/`   
- `notebooks/`          (for reproducing figures)  
  - `main/`  
  - `taiwanvvm/` 
## Notice
### Data
- Larger processed files can be accessed at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17199183.svg)](https://doi.org/10.5281/zenodo.17199183). Please place them under `/data/processed` after download. These include:
  - TaiESM1 daily IVT (historical)
  - TaiESM1 daily IVT (ssp585)
  - ERA5 summer IVT 2001–2019 (merged into a single file, used to train the VAE) 
### Environments
- `main` (`env_main.yml`):
  - The package of scientific colour map can be accessed at [this website](https://www.fabiocrameri.ch/colourmaps/) and refer to [Crameri et al. (2020)](https://www.nature.com/articles/s41467-020-19160-7).
- `taiwanvvm` (`env_taiwanvvm.yml`):
  - The package of `vvmtools` can be accessed and is documented at [GitHub](https://github.com/Aaron-Hsieh-0129/VVMTools).
