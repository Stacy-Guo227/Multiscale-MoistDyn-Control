# Multiscale-MoistDyn-Control

This repository contains code and resources for the paper:

> Guo et al. (2025). "A Promising Downscaling Strategy for Topographic Heavy Rainfalls over the Asian-Australian Monsoon Region by Leveraging Mutli-Scale Moisture Dynamical Control" *(Submitted)*.

## Structure
Multiscale-MoistDyn-Control/
├── README.md
├── environments/
│   └── env_main.yml      # (used for code in `notebooks/main` and `src/main`)
│   └── env_taiwanvvm.yml # (used for code in `notebooks/taiwanvvm` and `src/taiwanvvm`)
│   └── env_vae.yml       # (used for code in `src/vae/`)
├── data/
│   ├── raw/              # (unprocessed data and public URLs)
│   └── processed/        # (ready-to-use data)
│   └── vae_model/        # (model.h5, training log)
|       └── model_dataset # (dataset[train/val/test], fixed latent vectors)
├── notebooks/            # (for reproducing figures)
│   └── main/
│   └── taiwanvvm/
├── src/                  # (source code for processing data, model training, and plotting utilities)
│   └── main/
│   └── taiwanvvm/
└── └── vae/

## Notice
### `/data/`
- The following files are archived at doi:10.5281/zenodo.17199183 due to file sizes. Please place them under `/data/processed` after download.
  - TaiESM1 IVT (historical & SSP5-8.5)
  - ERA5 summer IVT 2001–2019 (single file, used to train the VAE) 
### `/environments/`
- `main` (`env_main.yml`):
  - The package of scientific colour map can be accessed at [Website](https://www.fabiocrameri.ch/colourmaps/) and refer to Crameri et al., 2020 [DOI](doi:10.1038/s41467-020-19160-7).
- `taiwanvvm` (`env_taiwanvvm.yml`):
  - The package of `vvmtools` can be accessed and is documented at [GitHub](https://github.com/Aaron-Hsieh-0129/VVMTools).
- 
