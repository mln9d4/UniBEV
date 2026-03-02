from mmdet.models import HEADS
from mmcv.runner import BaseModule
import torch
import torch.nn as nn
import os
from numpy import mean, var
import numpy as np
from tqdm import tqdm
import math
import torch
from torch import nn
from inspect import isfunction
from functools import partial
from einops import rearrange
import einops
import matplotlib.pyplot as plt

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        eps=1e-5
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        
        self.image_size = image_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn, eps=eps))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True, eps=eps),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False, eps=eps)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn, eps=eps))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups, eps=eps)

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0, eps=1e-5):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim, eps=eps),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32, eps=1e-5):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups, eps=eps)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout, eps=eps)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32, eps=1e-5):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel, eps=eps)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False, eps=1e-5):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout, eps=eps)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups, eps=eps)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    """
    Create a beta schedule that is a function of the number of diffusion steps.
    Return:
        betas: a numpy array of shape (n_timestep,) that defines the beta schedule
    """
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return torch.from_numpy(betas) if type(betas) == np.ndarray else betas

@HEADS.register_module()
class DenoiseDiffusion(BaseModule):
    def __init__(self, model_config, diffusion_config):
        super().__init__()
        # Parameters for training
        self.loss_fn = diffusion_config['loss_fn'] if diffusion_config['loss_fn'] is not None else nn.L1Loss()
        self.eps_model = UNet(**model_config)
        self.beta_schedule = diffusion_config['beta_schedule']
        self.k_noises = diffusion_config['k_noises']

        self.rolling_stats_img = RollingStatistics(channels=model_config['out_channel'])
        self.rolling_stats_pts = RollingStatistics(channels=model_config['out_channel'])

        # Parameters for diffusion process
        self.set_new_noise_schedule()         

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        self.n_steps = self.beta_schedule[phase]['n_timestep']
        to_torch = partial(torch.as_tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(**self.beta_schedule[phase])
        # self.betas = beta.type(dtype=torch.float32).to(self.eps_model.device)
        alphas = 1. - betas
        gammas = torch.cumprod(alphas, dim=0)
        sigmas = torch.sqrt(1.0 - gammas) # alpha_t in dpm-solver is gammas
        lambdas = torch.log(alphas / sigmas)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("gammas", to_torch(gammas))
        self.register_buffer("sigmas", to_torch(sigmas))
        self.register_buffer("lambdas", to_torch(lambdas))

    def save_output_img(self, bev_consumer_prediction, pts_bev_embed, epoch):
        save_dir = '/home/mingdayang/mmdetection3d/figures/diffusion_output_imgs/'
        os.makedirs(save_dir, exist_ok=True)
        bev_consumer_prediction = bev_consumer_prediction.cpu().numpy()
        pts_bev_embed = pts_bev_embed.cpu().numpy()
        # pts_bev_embed = rearrange(pts_bev_embed, 'b (h w) c -> b c h w', h=200, w=200).cpu().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        im0 = ax[0].imshow(bev_consumer_prediction[0, 0], cmap='viridis')
        ax[0].set_title('Prediction')
        im1 = ax[1].imshow(pts_bev_embed[0, 0], cmap='viridis')
        ax[1].set_title('Ground Truth')

        fig.colorbar(im0, ax=ax[0])
        fig.colorbar(im1, ax=ax[1]) 
        
        plt.savefig(os.path.join(save_dir, f'diffusion_output{epoch}.png'))
        plt.close(fig)

    def gather(self, tensor, t):
        """
        Gather the values of x at the time steps t.
        Makes it compatible with the shape of x0, which is (B, C, H, W).
        Args:
            tensor: a tensor of shape (n_steps,)
            t: a tensor of shape (B,)
        Return:
            a tensor of shape (B, 1, 1, 1) that contains the values of x at the time steps t
        """
        t = tensor.gather(-1, t)
        return t.reshape(-1, 1, 1, 1)
    
        # We need a function that samples the batch 
    def q_sample(self, y0, sample_gammas, noise=None):
        """
        Sample from q(yt|y0), reading same as sample xt at step t given x0.
        Other implementations also use function q_xt_x0 first but we can directly implement it here.
        Args:
            y0: the original data, shape (B, C, H, W)
            sample_gammas: the gamma values for sampling, shape (B,)
            noise: the noise, shape (B, C, H, W)
        Return:
            yt: the noisy data at time step t, shape (B, C, H, W)
        """
        eps = torch.randn_like(y0, device=y0.device) if noise is None else noise
        
        return (
            torch.sqrt(sample_gammas) * y0 + torch.sqrt(1 - sample_gammas) * eps
        )
    
    def knn_forward(self, y0, y_cond=None):
        """
        Algorithm 1 of the paper https://arxiv.org/pdf/2505.18521
        Instead of sampling randomly (miscible diffusion) we sample k random noise poinst and pick the knn nearest L2 dist and noise the image that way.
        Making our diffusion immiscible and accelerate training speed.

        Args:
            y0: the ground truth (B, C, H, W)
            y_cond: the input / condition (B, C, H, W)
        """
        print("Using knn forward for training")
        b, *_ = y0.shape
        
        ### START OF MMISCIBLE DIFFUSION ###
        
        # First generate n noises for each data in y0
        noise = torch.randn(y0.shape[0], self.k_noises, y0.shape[1], y0.shape[2], y0.shape[3], device=y0.device) # (B, k, C, H, W)

        y0_points = einops.rearrange(y0, 'b c h w -> b (h w c)')
        noise_points = einops.rearrange(noise, 'b k c h w -> b k (h w c)')
        
        # Calculate the distance between y0 and corresponding k noises
        distance_points = y0_points.unsqueeze(1) - noise_points # (B, k, D=H*W*C)
        distance = torch.linalg.vector_norm(distance_points, dim=2) # (B, k)
        
        # Pick the nearest noise for each data in latents
        _, min_index = torch.min(distance, dim=1) # (B)
        print(min_index)
        noise = torch.gather(noise, 1, min_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, y0.shape[1], y0.shape[2], y0.shape[3]))[:, 0, :, :, :] # (B, C, H, W)
        ### END OF IMMISCIBLE DIFFUSION ###

        t = torch.randint(1, self.n_steps, (b,), device=y0.device, dtype=torch.long)
        # Select a random gamma for each sample in the batch, which is between gamma_t and gamma_t-1 of generated timesteps t. This is to make the training more stable and avoid overfitting to specific timesteps.
        gamma_t1 = self.gather(self.gammas, t - 1)
        gamma_t2 = self.gather(self.gammas, t)
        sample_gammas = (gamma_t2 - gamma_t1) * torch.rand((b, 1, 1, 1), device=y0.device) + gamma_t1
        
        y_noisy = self.q_sample(y0, sample_gammas, noise=noise)

        noise_hat = self.eps_model(torch.cat([y_noisy, y_cond], dim=1) if y_cond is not None else y_noisy, sample_gammas)

        loss = self.loss_fn(noise_hat, noise)
        return loss


    def forward(self, y0, y_cond=None):
        """
        Algorithm 1 in Denoising Diffusion Probalisitic Models

        Args:
            y0: the original data, shape (B, C, H, W)
        """
        b, *_ = y0.shape

        t = torch.randint(1, self.n_steps, (b,), device=y0.device, dtype=torch.long)
        # Select a random gamma for each sample in the batch, which is between gamma_t and gamma_t-1 of generated timesteps t. This is to make the training more stable and avoid overfitting to specific timesteps.
        gamma_t1 = self.gather(self.gammas, t - 1)
        gamma_t2 = self.gather(self.gammas, t)
        sample_gammas = (gamma_t2 - gamma_t1) * torch.rand((b, 1, 1, 1), device=y0.device) + gamma_t1

        # Create the noise to compare it to the predicted noise, which is used for training the model. This is the noise added to the original data to get the noisy data at time step t.
        noise = torch.randn_like(y0, device=y0.device)
        y_noisy = self.q_sample(y0, sample_gammas, noise=noise)

        noise_hat = self.eps_model(torch.cat([y_noisy, y_cond], dim=1) if y_cond is not None else y_noisy, sample_gammas)

        loss = self.loss_fn(noise_hat, noise)
        return loss
    

    @torch.no_grad()
    def ddpm_sampler(self, y_cond=None, noise=None, sample_inter=10, clip_denoised=True):
        """
        https://arxiv.org/pdf/2006.11239
        Implementation of algorithm 2. However, to keep sampling stable we calculate the start from noise, clamp it and use the posterior of equation 7 to calculate y_t-1.
        We use equation 15 to calculate y_0 (start from noise), then clamp it. Then we use equation 7 to calculate y_t-1 = mean + sigma * z
        """
        y = torch.randn_like(y_cond, device=y_cond.device) if noise is None else noise
        ret_arr = y.clone()
        for i in tqdm(reversed(range(self.n_steps)), desc='DDPM sampler', total=self.n_steps):
            z = torch.randn_like(y) if i > 1 else torch.zeros_like(y)   
            t_tensor = torch.full((y_cond.shape[0],), i, device=y_cond.device, dtype=torch.long)

            gamma_t = self.gather(self.gammas, t_tensor)
            gamma_t_prev = self.gather(self.gammas, t_tensor - 1) if i > 0 else torch.ones_like(gamma_t)
            beta_t = self.gather(self.betas, t_tensor)
            alpha_t = self.gather(self.alphas, t_tensor)

            y_0_tilde = (y - torch.sqrt(1-gamma_t)*self.eps_model(torch.cat([y, y_cond], dim=1) if y_cond is not None else y, gamma_t)) / torch.sqrt(gamma_t) # predict start

            if clip_denoised:
                y_0_tilde = torch.clamp(y_0_tilde, -3., 3.)

            # eta = (y - torch.sqrt(alpha_t)*y_0_tilde) / torch.sqrt(1-alpha_t)

            # Calculate posterior mean and variance
            mean = (torch.sqrt(gamma_t_prev) * beta_t * y_0_tilde) / (1 - gamma_t) + (torch.sqrt(alpha_t) * (1 - gamma_t_prev) * y) / (1 - gamma_t)

            sigma = (1 - gamma_t_prev) * beta_t / (1 - gamma_t)

            y = mean + sigma * z

            if i % sample_inter == 0:
                ret_arr = torch.cat((ret_arr, y), dim=0)

        return y, ret_arr
    

    @torch.no_grad()
    def ddim_sampler(self, y_cond=None, noise=None, sample_inter=1, steps=50, clip_denoised=True, eta=0.0):
        """
        DDIM sampler from https://arxiv.org/abs/2010.02502
        With eta=0, it becomes a deterministic sampler, which is the one we will use in this implementation. With eta>0, it becomes a stochastic sampler, which is similar to the DDPM sampler but with different noise scale. 
        """
        y = torch.randn_like(y_cond, device=y_cond.device) if noise is None else noise
        ret_arr = y.clone()
        step_size = self.n_steps // steps
        time_steps = np.asarray(list(range(0, self.n_steps, step_size)))
        time_steps = np.linspace(0, self.n_steps-1, steps, dtype=int)
        time_steps_prev = np.concatenate(([0], time_steps[:-1]))
        print(time_steps)
        for i in tqdm(reversed(range(0, steps)), desc='DDIM sampling loop timestep', total=steps):
            # t = self.n_steps - i * step_size
            # t = max(t, 1)  # Ensure t does not go below 1
            # print(i)
            t_tensor = torch.full((y_cond.shape[0],), time_steps[i], dtype=torch.long, device=y_cond.device)
            # print(t_tensor)
            t_tensor_prev = torch.full((y_cond.shape[0],), time_steps_prev[i], dtype=torch.long, device=y_cond.device)
            # print(t_tensor_prev)
            gamma = self.gather(self.gammas, t_tensor)
            
            # Make sure that when t_tensor - step_size - 1 is negative, we use gamma_prev = 1, which means that we are at the final step and we should not add any noise.
            gamma_prev = self.gather(self.gammas, torch.clamp(t_tensor_prev, min=0)) # if (t_tensor_prev >= 0).any() else torch.ones_like(gamma)
            
            noise_pred = self.eps_model(torch.cat([y, y_cond], dim=1) if y_cond is not None else y, gamma)

            y0_pred = (y - torch.sqrt(1 - gamma) * noise_pred) / torch.sqrt(gamma)
            
            # Clamp prediction to stablize sampling
            if clip_denoised:
                y0_pred = torch.clamp(y0_pred, -3., 3.)

            sigma_t = eta * torch.sqrt((1 - gamma_prev) / (1 - gamma)) * torch.sqrt(1-gamma / gamma_prev)

            dir_yt = torch.sqrt(1 - gamma_prev - torch.pow(sigma_t, 2)) * noise_pred
            
            
            y = torch.sqrt(gamma_prev) * y0_pred + dir_yt + sigma_t * torch.randn_like(y)

            if i % sample_inter == 0:
                ret_arr = torch.cat((ret_arr, y), dim=0)

        return y, ret_arr
    

    @torch.no_grad()
    def dpm_solver_multi_step_sampler(self, y_cond=None, noise=None, sample_inter=10, steps=10, clip_denoised=True):
        """
        Implement multistep from https://arxiv.org/pdf/2211.01095
        """
        step_size = self.n_steps // steps
        time_steps = np.linspace(self.n_steps - 1, 0, steps+1, dtype=int)
        time_steps_prev = np.concatenate((time_steps[:-1], [0]))
        print(time_steps.shape)
        print(time_steps_prev.shape)
        print(time_steps)
        print(time_steps_prev)
        # time_steps = np.asarray(list(range(0, self.n_steps, step_size)))
        # time_steps_prev = np.concatenate(([0], time_steps[:-1]))


        yT = torch.randn_like(y_cond, device=y_cond.device) if noise is None else noise
        ret_arr = yT.clone()

        ytilde = yT
        
        t_start = torch.full((y_cond.shape[0],), time_steps[0], device=y_cond.device, dtype=torch.long)
        t_end = torch.full((y_cond.shape[0],), time_steps[1], device=y_cond.device, dtype=torch.long)

        lambda_start = self.gather(self.lambdas, t_start)
        lambda_end = self.gather(self.lambdas, t_end)
        h_prev = lambda_end - lambda_start # Positive step in log-SNR

         # Initial prediction at t_N (P)
        P = self.data_prediction(ytilde, y_cond=y_cond, t=t_start)
        if clip_denoised: P = torch.clamp(P, -3., 3.)

        """
        t_0 = torch.full((y_cond.shape[0],), time_steps[-1], device=y_cond.device, dtype=torch.long)
        t_1 = torch.full((y_cond.shape[0],), time_steps[-2], device=y_cond.device, dtype=torch.long)
        t_2 = torch.full((y_cond.shape[0],), time_steps[-3], device=y_cond.device, dtype=torch.long)
        # print(t_1)
        # print(t_0)
        h_i_prev = self.gather(self.lambdas, t_1) - self.gather(self.lambdas, t_0)

        # Buffer P and Q for multi_step sampling. P = -2 and Q = -1 in timestepe space, which means that they are the data prediction at t_i-2 and t_i-1 respectively. We will update them in each step and use them to calculate the data prediction at t_i.
        P = self.data_prediction(ytilde, y_cond=y_cond, t=t_0) # y_theta_0
        if clip_denoised:
            P = torch.clamp(P, -3., 3.)
        y_tilde = (self.gather(self.sigmas, t_1)/self.gather(self.sigmas, t_0)) * ytilde - self.gather(self.alphas, t_1) * (torch.exp(-h_i_prev) - 1) * P
        Q = self.data_prediction(y_tilde, y_cond=y_cond, t=t_1) # y_theta_2
        if clip_denoised:
            Q = torch.clamp(Q, -3., 3.)
        """
        # 1st order update: y_1 from y_0
        alpha_end = self.gather(self.alphas, t_end)
        sigma_start = self.gather(self.sigmas, t_start)
        sigma_end = self.gather(self.sigmas, t_end)
        
        ytilde = (sigma_end / sigma_start) * ytilde - alpha_end * (torch.exp(-h_prev) - 1) * P

        for i in tqdm((range(1, steps)), desc='DPM-Solver++(2M) sampler', initial=1, total=steps):
            print(i)
            # t_cur = self.n_steps - i * step_size - 1
            # t_prev = self.n_steps - (i - 1) * step_size - 1

            t_prev_tensor = torch.full((y_cond.shape[0],), time_steps_prev[i+1], device=y_cond.device, dtype=torch.long)
            t_cur_tensor = torch.full((y_cond.shape[0],), time_steps[i], device=y_cond.device, dtype=torch.long)
            # print(f"t_cur_tensor: {t_cur_tensor}")
            
            # Get model prediction at current state (Q)
            Q = self.data_prediction(ytilde, y_cond=y_cond, t=t_cur_tensor)
            if clip_denoised: Q = torch.clamp(Q, -3., 3.)

            h_cur = self.gather(self.lambdas, t_cur_tensor) - self.gather(self.lambdas, t_prev_tensor)
            print(f"h_i_cur: {h_cur}")
            r_i = h_prev / h_cur

            D_i = (1 + 1 / (2 * r_i)) * Q - 1 / (2 * r_i) * P

            alpha_next = self.gather(self.alphas, t_cur_tensor)
            sigma_cur = self.gather(self.sigmas, t_cur_tensor)
            sigma_next = self.gather(self.sigmas, t_prev_tensor)
            
            ytilde = (sigma_next / sigma_cur) * ytilde - alpha_next * (torch.exp(-h_cur) - 1) * D_i
            h_prev = h_cur
            P = Q.clone()


            if i % sample_inter == 0:
                ret_arr = torch.cat((ret_arr, ytilde), dim=0)

        return ytilde, ret_arr
    
    def data_prediction(self, yt, y_cond=None, t=None):
        gamma = self.gather(self.gammas, t).to(yt.device)
        noise_pred = self.eps_model(torch.cat([yt, y_cond], dim=1) if y_cond is not None else yt, gamma)
        y0_hat = (yt - torch.sqrt(1 - gamma) * noise_pred) / torch.sqrt(gamma)

        return y0_hat

class RollingStatistics(nn.Module):
    """
    Class for capturing the rolling statistics when doing online training. Calculates the mean and variance of the data per batch and updates the overall mean and variance using https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html formula.

    For online training only do it for one epoch after we have the overall statistics, then we can fix the statistics and do normalization and denormalization using the fixed statistics. This is because the statistics will not change much after one epoch of training, and it will be more stable to use fixed statistics for normalization and denormalization during training.

    Attributes:
        mean: The rolling mean of the data.
        var: The rolling variance of the data.
        std: The rolling standard deviation of the data.
        p_samples: The total number of observations seen so far. Used for calculating the new mean and variance when a new batch of data is observed. p_samples is the amount of samples in a batch.
    """
    def __init__(self, mean=None, var=None, std=None, p_samples=0, channels=256):
        super().__init__()
        self.fixed = False
        # self.mean = mean
        # self.var = var
        # self.std = std
        self.p_samples = p_samples
        self.channels = channels

        # register_buffer ensures these move with the model and save in state_dict
        self.register_buffer('mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('var', torch.ones(1, channels, 1, 1))
        self.register_buffer('std', torch.ones(1, channels, 1, 1))
        # self.register_buffer('p_samples', torch.tensor(0, dtype=torch.long))

    # def set_device(self, device):
    #     if self.mean is not None:
    #         self.mean = self.mean.to(device)
    #     if self.var is not None:
    #         self.var = self.var.to(device)
    #     if self.std is not None:
    #         self.std = self.std.to(device)

    def __str__(self):
        return f"RollingStatistics(mean={torch.mean(self.mean)}, var={torch.mean(self.var)}, std={torch.mean(self.std)}, p_samples={self.p_samples})"

    def update(self, batch_data):
        """
        batch_data shape: (B, C, H, W)
        """
        if self.fixed:
            return

        # Calculate current batch stats
        dims = (0, 2, 3)
        batch_mean = torch.mean(batch_data, dim=dims, keepdim=True)
        batch_var = torch.var(batch_data, dim=dims, keepdim=True, correction=0)
        
        # Calculate q_samples based on ALL elements reduced (B * H * W)
        q_samples = batch_data.shape[0] * batch_data.shape[2] * batch_data.shape[3]

        if self.p_samples == 0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var)
            self.p_samples = q_samples
        else:
            n_p = self.p_samples
            n_q = q_samples
            n_total = n_p + n_q

            # Welford's Parallel Mean
            new_mean = (n_p * self.mean + n_q * batch_mean) / n_total

            # Welford's Parallel Variance
            # Formula: [(n_p * var_p + n_q * var_q) / n_total] + [n_p * n_q * (mean_p - mean_q)^2 / n_total^2]
            mean_diff = self.mean - batch_mean
            new_var = ((n_p * self.var + n_q * batch_var) / n_total) + \
                      ((n_p * n_q) * torch.pow(mean_diff, 2) / (n_total ** 2))

            self.mean.copy_(new_mean)
            self.var.copy_(new_var)
            self.p_samples = n_total

        self.std.copy_(torch.sqrt(self.var + 1e-8))

    def normalize(self, sample):
        return (sample - self.mean) / self.std

    def denormalize(self, sample):
        return (sample * self.std) + self.mean

    def get_stats(self):
        return {'mean': self.mean, 'var': self.var, 'std': self.std, 'p_samples': self.p_samples, 'channels': self.channels}