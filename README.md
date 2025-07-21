# Mean Flows for One-step Generative Modeling

<div align="center">
    <img src="./assets/velocity.png" width="1000" alt="1-NFE sample with MeanFlow.">
</div>

<div align="center">
    <img src="./assets/samples.png" width="1000" alt="1-NFE sample with MeanFlow.">
</div>

This is the PyTorch re-implementation for the paper [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447). This code is based on the official [JAX implementation](https://github.com/Gsunshine/meanflow) and [REPA](https://github.com/sihyun-yu/REPA).

### 1. Environment setup

```bash
conda create -n meanflow python=3.10 -y
conda activate meanflow
pip install -r requirements.txt
```
### 2. Evaluation

You can generate images (and the .npz file can be used for [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations) suite) through the following script:

```bash
torchrun --nnodes=1 --nproc_per_node=8 generate_meanflow.py \
  --model SiT-B/4 \
  --num-fid-samples 50000 \
  --ckpt [YOUR_CHECKPOINT_PATH] \
  --per-proc-batch-size=64 \
  --vae=ema \
  --num-steps=1
```

The official repository provides a JAX checkpoint for `SiT-B/4`. I have converted it into a PyTorch checkpoint, which you can download [here](https://drive.google.com/file/d/1jcsM02gvPWe0IkXkZhDmLk6bIucsJG-b/view?usp=sharing). You can set `[YOUR_CHECKPOINT_PATH]` to the path of the downloaded `meanflow-B4.pth` and evaluate this checkpoint with the command above.

After obtaining the `.npz` result, you may first create a new conda environment to avoid conflicts following [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations) and download its `VIRTUAL_imagenet256_labeled.npz`. Then you could run the following command to get the metrics:

```bash
# in your new eval enviornment
python evaluator.py [YOUR_PATH_TO_VIRTUAL_imagenet256_labeled.npz] [YOUR_RESULT_npz]
```

### 3. Dataset

#### Dataset download

Currently, we provide experiments for [ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data). You can place the data that you want and can specifiy it via `--data-dir` arguments in training scripts. Please refer to our [preprocessing guide](preprocessing/).

### 4. Training

Example command:
```bash
accelerate launch train_meanflow.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="bf16" \
    --seed=0 \
    --model="SiT-XL/2" \
    --proj-coeff=0.0 \
    --encoder-depth=0 \
    --output-dir="exps" \
    --exp-name="meanflow-sitxl" \
    --batch-size=256 \
    --adam-beta2 0.95 \
    --epochs 240 \
    --gradient-accumulation-steps 2 \
    --t-start 0.0 \
    --t-end 0.75 \
    --omega 0.2 \
    --kappa 0.92 \
    --data-dir=[YOUR_DATA_PATH]
```

Then this script will automatically create the folder in `exps` to save logs and checkpoints. You can adjust the following options:

- `--models`: `[SiT-B/4, SiT-B/2, SiT-L/2, SiT-XL/2]`
- `--output-dir`: Any directory that you want to save checkpoints and logs
- `--exp-name`: Any string name (the folder will be created under `output-dir`)

> Warning: This repository is forked from REPA, and I keep some REPA options (such as `proj-coeff` and `encoder-depth`). However, they are actually not implemented and not supported yet. Just always disable them.

> Note: The `batch-size` option specifies the global batch size distributed across all devices. The actual local batch size on each GPU is calculated as `batch-size // num-gpus // gradient-accumulation-steps`.

## Note:

- I have made this repository executable, but I have not yet trained and evaluated it with the exact settings from the original paper to see if the performance matches. If you find any mismatches or implementation errors, or if you use this repository to reproduce the original paper's results, feel free to let me know!

- Due to the incompatibility between the Jacobian-vector product (jvp) operation and FlashAttention, the `fused_attn` flag should always be disabled for training. For evaluation, the flag can be enabled.

## Acknowledgement

This code is mainly built upon [REPA](https://github.com/sihyun-yu/REPA) and the official JAX implementation of [MeanFlow](https://github.com/Gsunshine/meanflow).

## BibTeX

```bibtex
@article{meanflow,
  title={Mean Flows for One-step Generative Modeling},
  author={Geng, Zhengyang and Deng, Mingyang and Bai, Xingjian and Kolter, J Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}
```
