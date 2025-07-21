from functools import partial
import torch
import torch.nn as nn


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class MeanFlow:
    """
    MeanFlow loss calculator, designed as a standalone class like a loss function.
    It is NOT a torch.nn.Module.
    """
    def __init__(
        self,
        # Noise distribution parameters
        noise_dist: str = 'logit_normal',
        P_mean: float = -0.4,
        P_std: float = 1.0,
        # Loss and Guidance parameters
        data_proportion: float = 0.75,
        guidance_eq: str = 'cfg',
        omega: float = 1.0,
        kappa: float = 0.5,
        t_start: float = 0.0,
        t_end: float = 1.0,
        jvp_fn='func',
        # Training dynamics parameters
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        # Model-related parameters (needed for guidance)
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        # Inference parameters (used by the standalone generate function)
        sampling_schedule_type: str = 'default'
    ):
        
        # Store all hyperparameters
        self.noise_dist = noise_dist
        self.P_mean = P_mean
        self.P_std = P_std
        self.data_proportion = data_proportion
        self.guidance_eq = guidance_eq
        self.omega = omega
        self.kappa = kappa
        self.t_start = t_start
        self.t_end = t_end
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.sampling_schedule_type = sampling_schedule_type
        self.dtype = torch.float32
        
        assert jvp_fn in ['func', 'autograd'], "jvp_fn must be 'func' or 'autograd'"
        if jvp_fn == 'func':
            self.jvp_fn = torch.func.jvp
        elif jvp_fn == 'autograd':
            self.jvp_fn = partial(torch.autograd.functional.jvp, create_graph=True)

    def _logit_normal_dist(self, bz, device):
        rnd_normal = torch.randn((bz, 1, 1, 1), dtype=self.dtype, device=device)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)
    
    def _uniform_dist(self, bz, device):
        return torch.rand((bz, 1, 1, 1), dtype=self.dtype, device=device)

    def sample_tr(self, bz, device):
        if self.noise_dist == 'logit_normal':
            t = self._logit_normal_dist(bz, device)
            r = self._logit_normal_dist(bz, device)
        elif self.noise_dist == 'uniform':
            t = self._uniform_dist(bz, device)
            r = self._uniform_dist(bz, device)
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")
        
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        data_size = int(bz * self.data_proportion)
        zero_mask = (torch.arange(bz, device=t.device) < data_size).view(bz, 1, 1, 1)
        r = torch.where(zero_mask, t, r)
        return t, r

    def u_fn(self, model, x, t, h, y, train=True):
        bz = x.shape[0]
        return model(x, t.reshape(bz), h.reshape(bz), y, train=train)
    
    def v_fn(self, model, x, t, y, train=False):
        h = torch.zeros_like(t)
        return self.u_fn(model, x, t, h, y=y, train=train)
    
    def guidance_fn(self, model, v_t, z_t, t, y, train=False):
        y_null = torch.full((z_t.shape[0],), self.num_classes, device=z_t.device)
        v_uncond = self.v_fn(model, z_t, t, y=y_null, train=train)

        if self.guidance_eq == 'cfg':
            omega_mask = (t >= self.t_start) & (t <= self.t_end)
            omega = torch.where(omega_mask, self.omega, 1.0)
            
            if self.kappa == 0:
                v_g = v_uncond + omega * (v_t - v_uncond)
            else: # kappa > 0
                v_cond = self.v_fn(model, z_t, t, y=y, train=train)
                kappa_mask = (t >= self.t_start) & (t <= self.t_end)
                kappa = torch.where(kappa_mask, self.kappa, 0.0)
                v_g   = omega * v_t + (1 - omega - kappa) * v_uncond + kappa * v_cond
        else:
            v_g = v_t
        
        return v_g

    def cond_drop(self, v_t, v_g, labels):
        bz = v_t.shape[0]
        rand_mask = torch.rand(labels.shape[0], device=labels.device) < self.class_dropout_prob
        num_drop = rand_mask.sum().int()
        drop_mask = torch.arange(bz, device=labels.device)[:, None, None, None] < num_drop
        
        y_inp = torch.where(drop_mask.reshape(bz,), self.num_classes, labels)
        v_g   = torch.where(drop_mask, v_t, v_g)
        return y_inp, v_g

    def __call__(self, model: nn.Module, imgs: torch.Tensor, labels: torch.Tensor, zs=None, train=True):
        """
        Calculates the training loss for MeanFlow.
        :param model: The neural network model (e.g., DiT/SiT).
        :param imgs: (B, C, H, W) tensor of images.
        :param labels: (B,) tensor of class labels.
        :return: A tuple of losses.
        """
        bz = imgs.shape[0]
        device = imgs.device
        x  = imgs.to(dtype=self.dtype)

        # -----------------------------------------------------------------
        # Instantaneous velocity
        t, r = self.sample_tr(bz, device)
        e = torch.randn_like(x)
        z_t = (1 - t) * x + t * e
        v = e - x
        
        # Guided velocity
        v_g = self.guidance_fn(model, v, z_t, t, labels, train=False) if self.guidance_eq == "cfg" else v
        
        # Cond dropout (dropout class labels)
        y_inp, v_g = self.cond_drop(v, v_g, labels)

        # -----------------------------------------------------------------
        # Compute u_tr (average velocity) and du_dt using jvp
        def u_fn(z_t, t, r):
            return self.u_fn(model, z_t, t, t - r, y=y_inp, train=train)

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(r)
        
        u, du_dt = self.jvp_fn(u_fn, (z_t, t, r), (v_g, dtdt, dtdr))

        # -----------------------------------------------------------------
        # Compute loss
        u_tgt = v_g - torch.clamp(t - r, min=0.0, max=1.0) * du_dt
        u_tgt = u_tgt.detach()
        
        denoising_loss = (u - u_tgt) ** 2
        denoising_loss = torch.sum(denoising_loss, dim=(1, 2, 3))

        # Adaptive weighting
        adp_wt = (denoising_loss + self.norm_eps) ** self.norm_p
        denoising_loss = denoising_loss / adp_wt.detach()
        
        # -----------------------------------------------------------------
        denoising_loss = denoising_loss.mean()  # mean over batch

        v_loss = torch.sum((u - v) ** 2, dim=(1, 2, 3)).mean().detach()
        
        # projection loss
        proj_loss = 0.
        if zs is not None and len(zs) > 0:
            bsz = zs[0].shape[0]
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            proj_loss /= (len(zs) * bsz)

        return denoising_loss, proj_loss, v_loss

    def sampling_schedule(self):
        if self.sampling_schedule_type == 'default':
            return torch.tensor([1.0, 0.0])
        else:
            raise ValueError(f"Unknown schedule: {self.sampling_schedule_type}")

    def solver_step(self, model, z_t, t, r, labels):
        u = self.u_fn(model, z_t, t=t, h=(t - r), y=labels)
        return z_t - (t - r).view(-1, 1, 1, 1) * u

    def sample_one_step(self, model, z_t, labels, i, t_steps):
        t = t_steps[i].expand(z_t.shape[0])
        r = t_steps[i + 1].expand(z_t.shape[0])
        return self.solver_step(model, z_t, t, r, labels)


@torch.no_grad()
def generate(
    mean_flow: MeanFlow, 
    model: nn.Module, 
    n_sample: int, 
    img_size: int, 
    img_channels: int, 
    num_classes: int, 
    device: str, 
    class_idx=None, 
    num_steps=1
):
    """
    Generate samples from the model using the MeanFlow sampling logic.
    :param mean_flow: An instance of the MeanFlow class.
    :param model: The trained DiT model.
    """
    model.eval()
    t_steps = mean_flow.sampling_schedule().to(device)

    x_shape = (n_sample, img_channels, img_size, img_size)
    z_t = torch.randn(x_shape, dtype=mean_flow.dtype, device=device)

    if class_idx is None:
        labels = torch.randint(0, num_classes, (n_sample,), device=device)
    else:
        labels = torch.full((n_sample,), class_idx, dtype=torch.long, device=device)

    for i in range(num_steps):
        z_t = mean_flow.sample_one_step(model, z_t, labels, i, t_steps)

    return z_t

@torch.no_grad()
def mean_flow_sampler(
    model: nn.Module,
    mean_flow,
    latents: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int = 1,
):
    model.eval()
    
    _dtype = latents.dtype
    x_next = latents.to(dtype=_dtype)
    device = x_next.device
    t_steps = mean_flow.sampling_schedule().to(device)
    
    for i in range(num_steps):
        x_next = mean_flow.sample_one_step(model, x_next, labels, i, t_steps)
    return x_next