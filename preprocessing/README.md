<h1 align="center"> Preprocessing Guide
</h1>

#### Dataset download

We follow the preprocessing code used in [edm2](https://github.com/NVlabs/edm2). In this code we made a several edits: (1) we removed unncessary parts except preprocessing because this code is only used for preprocessing, (2) we use [-1, 1] range for an input to the stable diffusion VAE (similar to DiT or SiT) unlike edm2 that uses [0, 1] range, and (3) we consider preprocessing to 256x256 resolution (or 512x512 resolution).

After downloading ImageNet, please run the following scripts (please update 256x256 to 512x512 if you want to do experiments on 512x512 resolution);

```bash
torchrun --nproc_per_node=8 preprocessing.py \
    --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=[TARGET_PATH]/vae-sd \
    --dest-images=[TARGET_PATH]/images \
    --batch-size=128 \
    --resolution=256 \
    --transform=center-crop-dhariwal
```


Here,`YOUR_DOWNLOAD_PATH` is the directory that you downloaded the dataset, and `TARGET_PATH` is the directory that you will save the preprocessed images and corresponding compressed latent vectors. This directory will be used for your experiment scripts. 

## Acknowledgement

The original code is mainly built upon [edm2](https://github.com/NVlabs/edm2) repository.

The multi-GPU preprocssing is writen by [this PR](https://github.com/sihyun-yu/REPA/pull/43).