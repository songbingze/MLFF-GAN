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

We conducted experiments using CIA and LGC data sets [[Emelyanova et al., 2013](https://www.sciencedirect.com/science/article/abs/pii/S0034425713000473)]. 

You can use the following links to download the CIA and LGC datasets:

CIA Dataset: http://dx.doi.org/10.4225/08/5111AC0BF1229

LGC Dataset: http://dx.doi.org/10.4225/08/5111AD2B7FEE6

The downloaded data needs to be converted to TIF and keeped the original file name, such as "L71093084_08420011007_HRF_modtran_surf_ref_agd66.tif", "MOD09GA_A2001281.sur_refl.tif". Next, organize the folders of the dataset into the following:

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

Trained model download: [[Link](https://1drv.ms/f/s!AtkYVZhw2KXSa1KSdtmLJtCWDdo?e=RKAmTN)]

The calculation of metrics in the paper, including all comparative experiments and ablation experiments models, utilized the 'skimage.metrics' module from the scikit-image library. 
The range of SSIM (Structural Similarity Index) is from -1.0 to 1.0 (the default setting in scikit-image from version 0.19.3 onward requires specifying the range parameter).

## Citation

If you find this work is useful for your research, please cite our paper:
```
@ARTICLE{9781347,
  author={Song, Bingze and Liu, Peng and Li, Jun and Wang, Lizhe and Zhang, Luo and He, Guojin and Chen, Lajiao and Liu, Jianbo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MLFF-GAN: A Multilevel Feature Fusion With GAN for Spatiotemporal Remote Sensing Images}, 
  year={2022},
  volume={60},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2022.3169916}}
```
