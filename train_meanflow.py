import argparse
from datetime import datetime
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict, defaultdict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit_meanflow import SiT_models
# from loss import SILoss
from utils import load_encoders
from meanflow import MeanFlow

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


# Plot per-parameter gradient distributions (histograms) and save under the experiment folder.
def plot_grad_distributions(
    grads,
    save_root,
    epoch,
    global_step,
    args,
    bins=100,
    max_panels_per_fig=12,
    param_names=None,
):
    """
    Plot per-parameter gradient distributions (histograms) and save under the experiment folder.

    Additionally dumps a JSON with per-parameter stats and simple group aggregates.
    """
    out_dir = os.path.join(save_root, "grads")
    os.makedirs(out_dir, exist_ok=True)

    # Format readable parameter display names using the last few tokens
    def format_name(pname: str, tail_parts: int = 3, max_chars: int = 40) -> str:
        tokens = str(pname).split(".")
        disp = ".".join(tokens[-min(tail_parts, len(tokens)):])
        if len(disp) > max_chars:
            keep = max_chars // 2 - 2
            disp = disp[:keep] + "..." + disp[-keep:]
        return disp

    # Flatten grads and compute stats
    flat_grads = []
    shapes = []
    stats = []
    for g in grads:
        if g is None:
            flat_grads.append(None)
            shapes.append(None)
            stats.append((float("nan"), float("nan"), float("nan")))
        else:
            g_np = g.detach().float().view(-1).cpu().numpy()
            flat_grads.append(g_np)
            shapes.append(tuple(g.size()))
            mu = float(np.mean(g_np)) if g_np.size > 0 else float("nan")
            sigma = float(np.std(g_np)) if g_np.size > 0 else float("nan")
            l2 = float(np.linalg.norm(g_np)) if g_np.size > 0 else float("nan")
            stats.append((mu, sigma, l2))

    def fig_title(base_idx, idx_end):
        return (
            f"Grad dists | exp={args.exp_name} | epoch={epoch} | step={global_step} | "
            f"bs={args.batch_size} | lr={args.learning_rate:g} | clip={args.max_grad_norm:g} | "
            f"range={base_idx}-{idx_end-1}"
        )

    n_params = len(flat_grads)
    saved_paths = []
    if n_params == 0:
        return saved_paths

    if param_names is None or len(param_names) != n_params:
        param_names = [f"param[{i}]" for i in range(n_params)]

    cols = 4
    rows = int(np.ceil(min(max_panels_per_fig, n_params) / cols))

    stats_all = []
    group_aggr = defaultdict(lambda: {"count": 0, "l2_sq_sum": 0.0})

    start = 0
    while start < n_params:
        end = min(start + max_panels_per_fig, n_params)
        chunk = flat_grads[start:end]
        chunk_stats = stats[start:end]
        chunk_names = param_names[start:end]
        chunk_shapes = shapes[start:end]

        panels = end - start
        rows = int(np.ceil(panels / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.0 * rows), squeeze=False)
        axes = axes.flatten()

        for i, (g_np, (mu, sigma, l2), pname, pshape) in enumerate(zip(chunk, chunk_stats, chunk_names, chunk_shapes)):
            ax = axes[i]
            if g_np is None or g_np.size == 0:
                disp = format_name(pname)
                ax.set_title(f"{disp} | empty | idx={start+i}")
                ax.axis("off")
                stats_all.append(
                    {
                        "param_index": int(start + i),
                        "name": str(pname),
                        "group": ".".join(str(pname).split(".")[:2]) if "." in str(pname) else str(pname),
                        "count": 0,
                        "mean": None,
                        "std": None,
                        "l2": None,
                        "min": None,
                        "max": None,
                        "p01": None,
                        "p999": None,
                        "hist": None,
                        "edges": None,
                    }
                )
                continue

            lo = np.percentile(g_np, 0.1)
            hi = np.percentile(g_np, 99.9)
            if lo == hi:
                lo, hi = lo - 1e-12, hi + 1e-12

            ax.hist(g_np, bins=bins, range=(lo, hi), log=True)
            disp = format_name(pname, tail_parts=3, max_chars=40)
            shape_str = "x".join(map(str, pshape)) if pshape is not None else "?"
            ax.set_title(f"{disp} | idx={start+i} | {shape_str}\nμ={mu:.2e} σ={sigma:.2e} ‖g‖₂={l2:.2e}")
            ax.set_xlabel("grad value")
            ax.set_ylabel("count")

            counts, edges = np.histogram(g_np, bins=bins, range=(lo, hi))
            stats_all.append(
                {
                    "param_index": int(start + i),
                    "name": str(pname),
                    "group": ".".join(str(pname).split(".")[:2]) if "." in str(pname) else str(pname),
                    "count": int(g_np.size),
                    "mean": float(mu),
                    "std": float(sigma),
                    "l2": float(l2),
                    "min": float(np.min(g_np)),
                    "max": float(np.max(g_np)),
                    "p01": float(lo),
                    "p999": float(hi),
                    "hist": counts.astype(int).tolist(),
                    "edges": edges.astype(float).tolist(),
                }
            )

            group_key = ".".join(str(pname).split(".")[:2]) if "." in str(pname) else str(pname)
            ga = group_aggr[group_key]
            ga["count"] += int(g_np.size)
            ga["l2_sq_sum"] += float(l2) ** 2

        # hide unused axes
        for j in range(len(chunk), len(axes)):
            axes[j].axis("off")

        fig.suptitle(fig_title(start, end), fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])

        fname = (
            f"epoch{epoch:04d}_step{global_step:07d}"
            f"_bs{args.batch_size}"
            f"_lr{args.learning_rate:g}"
            f"_clip{args.max_grad_norm:g}"
            f"_parts{start:04d}-{end-1:04d}.png"
        )
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        saved_paths.append(fpath)

        start = end

    try:
        meta = {
            "exp_name": args.exp_name,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "max_grad_norm": float(args.max_grad_norm),
            "num_params": int(n_params),
            "bins": int(bins),
        }
        groups_payload = []
        for gname, gvals in sorted(group_aggr.items()):
            groups_payload.append(
                {
                    "group": gname,
                    "count": int(gvals["count"]),
                    "l2": float(math.sqrt(gvals["l2_sq_sum"])) if gvals["l2_sq_sum"] > 0 else 0.0,
                }
            )

        stats_payload = {"meta": meta, "params": stats_all, "groups": groups_payload}
        stats_fname = (
            f"epoch{epoch:04d}_step{global_step:07d}"
            f"_bs{args.batch_size}"
            f"_lr{args.learning_rate:g}"
            f"_clip{args.max_grad_norm:g}_stats.json"
        )
        stats_fpath = os.path.join(out_dir, stats_fname)
        with open(stats_fpath, "w") as f:
            json.dump(stats_payload, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save gradient stats JSON: {e}")

    return saved_paths


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # Auto-append timestamp to experiment name (YYYYMMDDHHMM), skip when resuming
    try:
        if getattr(args, "resume_step", 0) == 0:
            ts = datetime.now().strftime("%Y%m%d%H%M")
            args.exp_name = f"{args.exp_name}-{ts}"
    except Exception:
        pass
    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.guidance_eq == "cfg"),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = MeanFlow(
        noise_dist=args.noise_dist,
        P_mean=args.P_mean,
        P_std=args.P_std,
        data_proportion=args.data_proportion,
        guidance_eq=args.guidance_eq,
        omega=args.omega,
        kappa=args.kappa,
        t_start=args.t_start,
        t_end=args.t_end,
        jvp_fn=args.jvp_fn,
        norm_p=args.norm_p,
        norm_eps=args.norm_eps,
        num_classes=args.num_classes,
        class_dropout_prob=args.class_dropout_prob,
        sampling_schedule_type=args.sampling_schedule_type,
        stop_gradient=args.stop_gradient,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes // args.gradient_accumulation_steps)
    
    steps_per_epoch = len(train_dataset) // args.batch_size
    max_train_steps = min(args.max_train_steps, steps_per_epoch * args.epochs)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Save an initial checkpoint at training start to verify saving works
    if accelerator.is_main_process:
        try:
            to_save_model = accelerator.unwrap_model(model)
            init_ckpt = {
                "model": to_save_model.state_dict(),
                "ema": ema.state_dict(),
                "opt": optimizer.state_dict(),
                "args": args,
                "steps": int(global_step),
            }
            init_ckpt_path = f"{checkpoint_dir}/{0:07d}.pt"
            if not os.path.exists(init_ckpt_path):
                accelerator.save(init_ckpt, init_ckpt_path)
                logger.info(f"Saved initial checkpoint to {init_ckpt_path}")
        except Exception as e:
            logger.warning(f"Failed to save initial checkpoint: {e}")

    # Labels to condition the model with (feel free to change):
    # Use the same per-process batch size for sampling as training to avoid OOM and keep behavior consistent
    sample_batch_size = local_batch_size
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
        
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            labels = y
            zs = None
            # Metrics placeholders per step
            was_clipped = 0.0
            grad_to_weight_ratio_mean = float("nan")
            grad_sparsity_pre = float("nan")
            grad_sparsity_post = float("nan")
            cos_sim_prev_grad = None
            prev_params_snapshot = None
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                if args.encoder_depth > 0:
                    zs = []
                    with accelerator.autocast():
                        for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                            raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                            z = encoder.forward_features(raw_image_)
                            if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                            if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                            zs.append(z)

            with accelerator.accumulate(model):
                # Mixed precision forward + loss for speed/VRAM
                with accelerator.autocast():
                    loss, proj_loss, v_loss = loss_fn(model, x, labels=labels, zs=zs)
                loss_mean = loss.mean()
                if args.encoder_depth > 0:
                    proj_loss_mean = proj_loss.mean()
                    loss = loss_mean + proj_loss_mean * args.proj_coeff
                else:
                    loss = loss_mean
                v_loss_mean = v_loss.detach().mean()
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Compute sparsity BEFORE clipping
                    preclip_grads = [
                        p.grad.detach().clone() if (p.requires_grad and p.grad is not None) else None
                        for p in model.parameters()
                    ]
                    total_elems_pre = 0
                    zero_elems_pre = 0
                    for g in preclip_grads:
                        if g is None:
                            continue
                        total_elems_pre += g.numel()
                        thr = 1e-8 if g.dtype in (torch.float16, torch.bfloat16) else 1e-12
                        zero_elems_pre += (g.abs() <= thr).sum().item()
                    if total_elems_pre > 0:
                        grad_sparsity_pre = float(zero_elems_pre) / float(total_elems_pre)

                    # Clip gradients (grad_norm returned is pre-clip global norm)
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                    # Collect gradients AFTER clipping for diagnostics
                    grads = [p.grad.detach().clone() if (p.requires_grad and p.grad is not None) else None for p in model.parameters()]

                    # Whether clipping actually happened
                    try:
                        was_clipped = float(grad_norm.item() > args.max_grad_norm + 1e-12)
                    except Exception:
                        was_clipped = 0.0

                    # Mean grad/weight ratio across parameters
                    gw_ratios = []
                    for p, g in zip(model.parameters(), grads):
                        if g is None:
                            continue
                        pw = p.data.float().norm().item()
                        gw = g.float().norm().item()
                        if pw > 0:
                            gw_ratios.append(gw / pw)
                    if len(gw_ratios) > 0:
                        grad_to_weight_ratio_mean = float(np.mean(gw_ratios))

                    # Gradient sparsity AFTER clipping (fraction of near-zero entries overall)
                    total_elems = 0
                    zero_elems = 0
                    for g in grads:
                        if g is None:
                            continue
                        total_elems += g.numel()
                        thr = 1e-8 if (g is not None and g.dtype in (torch.float16, torch.bfloat16)) else 1e-12
                        zero_elems += (g.abs() <= thr).sum().item()
                    if total_elems > 0:
                        grad_sparsity_post = float(zero_elems) / float(total_elems)

                    # Gradient cosine similarity with previous step (only when logging to reduce cost)
                    if accelerator.is_main_process and (global_step == 0 or (global_step % args.sampling_steps == 0)):
                        try:
                            flat_list = [g.view(-1) for g in grads if g is not None and g.numel() > 0]
                            if len(flat_list) > 0:
                                gflat = torch.cat(flat_list).float().cpu()
                                # init cache on first use
                                if not hasattr(main, "_prev_grad_flat"):
                                    main._prev_grad_flat = None
                                if main._prev_grad_flat is not None and main._prev_grad_flat.numel() == gflat.numel():
                                    denom = (main._prev_grad_flat.norm() * gflat.norm()).item() + 1e-12
                                    cos_sim_prev_grad = float(torch.dot(main._prev_grad_flat, gflat).item() / denom)
                                main._prev_grad_flat = gflat #这个main的用法无敌，可以方便的调用上一次记录的结果
                        except Exception:
                            cos_sim_prev_grad = None

                    # Plot and save gradient distributions at a reasonable interval for inspection
                    # study the gradient behavior of different layer
                    if accelerator.is_main_process and (global_step == 0 or (global_step % args.sampling_steps == 0)):
                        save_root = os.path.join(args.output_dir, args.exp_name)
                        try:
                            # Build param names aligned with model.parameters() order
                            param_names = [name for name, _ in model.named_parameters()]
                            saved_paths = plot_grad_distributions(
                                grads=grads,
                                save_root=save_root,
                                epoch=epoch,
                                global_step=global_step,
                                args=args,
                                bins=100,
                                max_panels_per_fig=24,
                                param_names=param_names,
                            )
                            # Log the last page image to wandb for quick glance
                            if len(saved_paths) > 0:
                                last_img_path = saved_paths[-1]
                                accelerator.log({"grad_distributions_last": wandb.Image(last_img_path)}, step=global_step)

                            # Upload the corresponding JSON stats file
                            try:
                                grads_dir = os.path.join(args.output_dir, args.exp_name, "grads")
                                stats_fname = (
                                    f"epoch{epoch:04d}_step{global_step:07d}"
                                    f"_bs{args.batch_size}"
                                    f"_lr{args.learning_rate:g}"
                                    f"_clip{args.max_grad_norm:g}_stats.json"
                                )
                                stats_fpath = os.path.join(grads_dir, stats_fname)
                                if os.path.exists(stats_fpath):
                                    try:
                                        artifact = wandb.Artifact(
                                            name=f"grad-stats-epoch{epoch:04d}-step{global_step:07d}",
                                            type="grad-stats",
                                        )
                                        artifact.add_file(stats_fpath)
                                        wandb.log_artifact(artifact)
                                    except Exception as _e:
                                        # Fallback: attach as a run file
                                        wandb.save(stats_fpath)
                            except Exception as e:
                                logging.warning(f"Failed to log gradient stats JSON to wandb: {e}")
                        except Exception as e:
                            logging.warning(f"plot_grad_distributions failed: {e}")
                
                
                
                
                
                # Take parameter snapshot before step only when we plan to log update size (to save memory)
                take_update_snapshot = accelerator.sync_gradients and accelerator.is_main_process and (global_step == 0 or (global_step % args.sampling_steps == 0))
                if take_update_snapshot:
                    prev_params_snapshot = [p.data.detach().clone() for p in model.parameters() if p.requires_grad]

                optimizer.step() # Accelerate will no-op on non-sync micro steps
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        try:
                            to_save_model = accelerator.unwrap_model(model)
                            checkpoint = {
                                "model": to_save_model.state_dict(),
                                "ema": ema.state_dict(),
                                "opt": optimizer.state_dict(),
                                "args": args,
                                "steps": int(global_step),
                            }
                            checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                            accelerator.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved checkpoint to {checkpoint_path}")
                        except Exception as e:
                            logger.warning(f"Failed to save checkpoint at step {global_step}: {e}")

                if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                    from meanflow import mean_flow_sampler
                    with torch.no_grad(), accelerator.autocast():
                        samples = mean_flow_sampler(
                            model, 
                            loss_fn,
                            xT, 
                            ys,
                            num_steps=1, 
                        )
                        # Decode under autocast; cast to fp32 for logging after gather
                        samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                        gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                        samples = (samples + 1) / 2.
                        gt_samples = (gt_samples + 1) / 2.
                    out_samples = accelerator.gather(samples.to(torch.float32))
                    gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                    accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                    "gt_samples": wandb.Image(array2grid(gt_samples))})
                    logging.info("Generating EMA samples done.")

                logs = {
                    "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                    "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                    "v_loss": accelerator.gather(v_loss_mean).mean().detach().item(),
                }
                # Extended stability diagnostics
                logs["was_clipped"] = was_clipped
                logs["grad_to_weight_ratio_mean"] = grad_to_weight_ratio_mean
                # Log both pre/post sparsity; keep legacy key mapped to post-clip
                logs["grad_sparsity_pre"] = grad_sparsity_pre
                logs["grad_sparsity_post"] = grad_sparsity_post
                logs["grad_sparsity"] = grad_sparsity_post

                # Compute single-step update norm and update/weight ratio if we took a snapshot
                if prev_params_snapshot is not None:
                    try:
                        delta_sq_sum = 0.0
                        upd_ratios = []
                        # zip aligns with requires_grad params order
                        for p, p_prev in zip(model.parameters(), prev_params_snapshot):
                            if not p.requires_grad:
                                continue
                            delta = (p.data - p_prev).float()
                            delta_sq_sum += delta.pow(2).sum().item()
                            pw = p_prev.float().norm().item()
                            dw = delta.norm().item()
                            if pw > 0:
                                upd_ratios.append(dw / pw)
                        logs["update_norm"] = float(math.sqrt(delta_sq_sum))
                        logs["update_to_weight_ratio_mean"] = float(np.mean(upd_ratios)) if len(upd_ratios) > 0 else float("nan")
                    except Exception:
                        pass

                # Optimizer moment norms (AdamW)
                try:
                    m2 = 0.0
                    v2 = 0.0
                    for group in optimizer.param_groups:
                        for p in group.get("params", []):
                            st = optimizer.state.get(p, None)
                            if not st:
                                continue
                            m = st.get("exp_avg", None)
                            v = st.get("exp_avg_sq", None)
                            if m is not None:
                                m2 += m.float().pow(2).sum().item()
                            if v is not None:
                                v2 += v.float().pow(2).sum().item()
                    logs["adam_m_norm"] = float(math.sqrt(m2)) if m2 > 0 else 0.0
                    logs["adam_v_sqrt_sum"] = float(math.sqrt(v2)) if v2 > 0 else 0.0
                except Exception:
                    pass

                # Optional: gradient cosine similarity to previous step
                if cos_sim_prev_grad is not None:
                    logs["cos_sim_prev_grad"] = cos_sim_prev_grad
                if args.encoder_depth > 0:
                    logs["proj_loss"] = accelerator.gather(proj_loss_mean).mean().detach().item()
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=0)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=20000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--noise-dist", type=str, default="logit_normal", choices=["uniform", "logit_normal"])
    parser.add_argument("--P-mean", type=float, default=-0.4)
    parser.add_argument("--P-std", type=float, default=1.0)
    parser.add_argument("--data-proportion", type=float, default=0.75)
    parser.add_argument("--guidance-eq", type=str, default="cfg")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--jvp-fn", type=str, default="func", choices=["func", "autograd"])
    parser.add_argument("--norm-p", type=float, default=1.0)
    parser.add_argument("--norm-eps", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--sampling-schedule-type", type=str, default="default")
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)

    parser.add_argument("--stop-gradient", action=argparse.BooleanOptionalAction, default=True)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    if args.fused_attn:
        raise NotImplementedError("fused (flash) attn is not compatible with jvp!")

    if args.encoder_depth > 0:
        raise NotImplementedError("MeanFlow with REPA is not implemented yet")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
