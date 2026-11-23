import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embedding untuk DDPM."""
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(Block, self).__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        h = self.relu(self.bn1(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        h = h + time_emb[(..., ) + (None, ) * 2]
        h = self.relu(self.bn2(self.conv2(h)))
        return h


class DiffusionUNet(nn.Module):
    """
    Lightweight UNet untuk citra 28x28 (Fashion-MNIST)
    Memiliki encoder-decoder dengan skip connection dan sinusoidal timestep conditioning.
    """
    def __init__(self, in_channels=1, time_emb_dim=64):
        super(DiffusionUNet, self).__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True)
        )
        
        # Encoder (Downsampling)
        self.inc = Block(in_channels, 32, time_emb_dim)
        self.down1 = nn.MaxPool2d(2)  # 28x28 -> 14x14
        self.conv_down1 = Block(32, 64, time_emb_dim)
        self.down2 = nn.MaxPool2d(2)  # 14x14 -> 7x7
        
        # Bottleneck
        self.bot = Block(64, 128, time_emb_dim)
        
        # Decoder (Upsampling)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')  # 7x7 -> 14x14
        self.conv_up1 = Block(128 + 64, 64, time_emb_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')  # 14x14 -> 28x28
        self.conv_up2 = Block(64 + 32, 32, time_emb_dim)
        
        self.outc = nn.Conv2d(32, in_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        
        # Down
        x1 = self.inc(x, t)
        x2 = self.down1(x1)
        x2 = self.conv_down1(x2, t)
        x3 = self.down2(x2)
        
        # Bottleneck
        x_bot = self.bot(x3, t)
        
        # Up with skips
        x_up = self.up1(x_bot)
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.conv_up1(x_up, t)
        
        x_up = self.up2(x_up)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.conv_up2(x_up, t)
        
        out = self.outc(x_up)
        return out


class GaussianDiffusion(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM)
    Sesuai Ho et al. (2020)
    """
    def __init__(self, model, timesteps=300, beta_start=1e-4, beta_end=0.02, device="cpu"):
        super(GaussianDiffusion, self).__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Koefisien untuk sampling forward q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Koefisien untuk sampling posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: menyuntikkan derau ke citra x_0 pada timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start):
        """Menghitung MSE noise prediction loss: L = E[||eps - eps_theta(x_t, t)||^2]."""
        batch_size = x_start.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t_index):
        """Satu langkah reverse denoising dari x_t menuju x_{t-1}."""
        t = torch.full((x.size(0),), t_index, device=self.device, dtype=torch.long)
        predicted_noise = self.model(x, t)
        
        beta = self.betas[t_index]
        alpha = self.alphas[t_index]
        alpha_cumprod = self.alphas_cumprod[t_index]
        
        # Rumus rata-rata posterior mu_theta(x_t, t)
        mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
        
        if t_index == 0:
            return mean
        else:
            posterior_var = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(self, shape, return_trajectory=False, stride=1):
        """
        Reverse diffusion bertahap x_T -> x_0
        Jika return_trajectory=True, mengembalikan list citra pada interval tertentu
        untuk visualisasi Figur 5.
        """
        self.model.eval()
        img = torch.randn(shape, device=self.device)
        trajectory = [img.clamp(0.0, 1.0).cpu()]
        
        for t in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, t)
            if return_trajectory and (t % stride == 0 or t == 0):
                trajectory.append(img.clamp(0.0, 1.0).cpu())
                
        img = img.clamp(0.0, 1.0)
        if return_trajectory:
            return img, trajectory
        return img
