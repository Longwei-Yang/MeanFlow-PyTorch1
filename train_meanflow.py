import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
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
    max_panels_per_fig=24,
):
    """
    Plot per-parameter gradient distributions (histograms) and save under the experiment folder.

    Args:
        grads (List[Tensor]): list of per-parameter gradients in the same order as model.parameters()
                              e.g., grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
        save_root (str): root dir for this experiment, e.g., os.path.join(args.output_dir, args.exp_name)
        epoch (int): current epoch index
        global_step (int): current global training step (before or after increment is fine, used for naming)
        args (argparse.Namespace): full args; we will extract key fields for filename to help searching
        bins (int): number of histogram bins
        max_panels_per_fig (int): maximum number of subplots per saved figure; if there are more parameters,
                                  the function will create multiple figures.
    Returns:
        List[str]: list of saved figure paths
    """
    # Ensure output directory
    out_dir = os.path.join(save_root, "grads")
    os.makedirs(out_dir, exist_ok=True)

    # Flatten grads and convert to numpy for plotting
    flat_grads = []
    stats = []
    for g in grads:
        if g is None:
            flat_grads.append(None)
            stats.append((float('nan'), float('nan'), float('nan')))
            continue
        g_np = g.detach().float().view(-1).cpu().numpy()
        flat_grads.append(g_np)
        mu = float(np.mean(g_np)) if g_np.size > 0 else float('nan')
        sigma = float(np.std(g_np)) if g_np.size > 0 else float('nan')
        l2 = float(np.linalg.norm(g_np)) if g_np.size > 0 else float('nan')
        stats.append((mu, sigma, l2))

    # Helper to build a readable, disambiguating title
    def fig_title(base_idx, idx_end):
        return (f"Grad dists | exp={args.exp_name} | epoch={epoch} | step={global_step} | "
                f"bs={args.batch_size} | lr={args.learning_rate:g} | "
                f"clip={args.max_grad_norm:g} | range={base_idx}-{idx_end-1}")

    # Number of figures needed
    n_params = len(flat_grads)
    saved_paths = []
    if n_params == 0:
        return saved_paths

    # Determine subplot grid (rows x cols) up to max_panels_per_fig, prefer 4 columns
    cols = 4
    rows = int(np.ceil(min(max_panels_per_fig, n_params) / cols))

    # Iterate chunks
    start = 0
    while start < n_params:
        end = min(start + max_panels_per_fig, n_params)
        chunk = flat_grads[start:end]
        chunk_stats = stats[start:end]

        # Recompute rows for last chunk if smaller
        panels = end - start
        rows = int(np.ceil(panels / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.0 * rows), squeeze=False)
        axes = axes.flatten()

        for i, (g_np, (mu, sigma, l2)) in enumerate(zip(chunk, chunk_stats)):
            ax = axes[i]
            if g_np is None or g_np.size == 0:
                ax.set_title(f"param[{start + i}] | empty")
                ax.axis("off")
                continue

            # Robust clipping of x-axis range for visibility
            # Use percentiles to ignore extreme outliers when drawing hist
            lo = np.percentile(g_np, 0.1)
            hi = np.percentile(g_np, 99.9)
            if lo == hi:
                lo, hi = lo - 1e-12, hi + 1e-12

            ax.hist(g_np, bins=bins, range=(lo, hi), log=True)
            ax.set_title(f"param[{start + i}] μ={mu:.2e} σ={sigma:.2e} ‖g‖₂={l2:.2e}")
            ax.set_xlabel("grad value")
            ax.set_ylabel("count")

        # Hide any unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Global title and tight layout
        fig.suptitle(fig_title(start, end), fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])

        # Build filename with key args for future search
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

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
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
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]

                # Plot and save gradient distributions at a reasonable interval for inspection
                # study the gradient behavior of different layer
                if accelerator.is_main_process and (global_step == 0 or (global_step % args.sampling_steps == 0)):
                    save_root = os.path.join(args.output_dir, args.exp_name)
                    try:
                        saved_paths = plot_grad_distributions(
                            grads=grads,
                            save_root=save_root,
                            epoch=epoch,
                            global_step=global_step,
                            args=args,
                            bins=100,
                            max_panels_per_fig=24,
                        )
                        # Optionally log first page to wandb for quick glance
                        if len(saved_paths) > 0:
                            accelerator.log({"grad_distributions": wandb.Image(saved_paths[0])}, step=global_step)
                    except Exception as e:
                        logging.warning(f"plot_grad_distributions failed: {e}")
                
                
                
                
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": optimizer.state_dict(),
                            "args": args,
                            "steps": global_step,
                        }
                        checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                    from meanflow import mean_flow_sampler
                    with torch.no_grad():
                        samples = mean_flow_sampler(
                            model, 
                            loss_fn,
                            xT, 
                            ys,
                            num_steps=1, 
                        ).to(torch.float32)
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
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
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
