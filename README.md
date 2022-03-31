# MLFF-GAN: A Multi-level Feature Fusion with GAN for Spatiotemporal Remote Sensing Images
This repository provides code for training and testing the models for spatiotemporal remote sensing images fusion with the official PyTorch implementation of the following paper:

> MLFF-GAN: A Multi-level Feature Fusion with GAN for Spatiotemporal Remote Sensing Images
> 
> Bingze Song, Peng Liu, Jun Li, Lizhe Wang, Luo Zhang, Guojin He, Lajiao Chen, Jianbo Liu

## Environment

This framework is built using Python 3.7.

The following command installs all necessary packages:

```
pip install -r requirements.txt
```

## Dataset

We conducted experiments using CIA and LGC data sets [[Emelyanova et al., 2013](https://www.sciencedirect.com/science/article/abs/pii/S0034425713000473)]. The downloaded data needs to be converted to TIF and keeped the original file name, such as "L71093084_08420011007_HRF_modtran_surf_ref_agd66.tif", "MOD09GA_A2001281.sur_refl.tif". Next, organize the folders of the dataset into the following:

```
dataset
├─train
│  ├─2001_281_1007-2001_290_1016
│  │      L71093084_08420011007_HRF_modtran_surf_ref_agd66.tif
│  │      L71093084_08420011016_HRF_modtran_surf_ref_agd66.tif
│  │      MOD09GA_A2001281.sur_refl.tif
│  │      MOD09GA_A2001290.sur_refl.tif
│  │
│  ├─...
│
└─val
    ├─2002_076_0316-2002_092_0401
    │      L71093084_08420020316_HRF_modtran_surf_ref_agd66.tif
    │      L71093084_08420020401_HRF_modtran_surf_ref_agd66.tif
    │      MOD09GA_A2002076.sur_refl.tif
    │      MOD09GA_A2002092.sur_refl.tif
    │
    └─...
```

## Train and test

You can run training and testing programs through the following commands:

```
python run.py --data_dir [dataset path] --save_dir [save path]
```
