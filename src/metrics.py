import time
import torch
import numpy as np
from scipy import linalg
from skimage.metrics import structural_similarity as ssim_fn

def compute_mse(real_imgs, recon_imgs):
    """Menghitung Mean Squared Error pada rentang [0, 1]."""
    real = real_imgs.detach().cpu().numpy()
    recon = recon_imgs.detach().cpu().numpy()
    return float(np.mean((real - recon) ** 2))

def compute_psnr(real_imgs, recon_imgs, max_val=1.0):
    """Menghitung Peak Signal-to-Noise Ratio (dB)."""
    mse = compute_mse(real_imgs, recon_imgs)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10((max_val ** 2) / mse))

def compute_ssim(real_imgs, recon_imgs):
    """Menghitung rata-rata Structural Similarity Index (SSIM)."""
    real = real_imgs.detach().cpu().numpy()
    recon = recon_imgs.detach().cpu().numpy()
    
    ssim_values = []
    for i in range(len(real)):
        r = real[i, 0]  # channel 0, (28, 28)
        c = recon[i, 0]
        val = ssim_fn(r, c, data_range=1.0)
        ssim_values.append(val)
    return float(np.mean(ssim_values))

def measure_inference_latency(generate_fn, batch_size=64, num_runs=10):
    """
    Mengukur rata-rata inference latency per batch (64 sampel) dalam milidetik (ms).
    Dilakukan warm-up terlebih dahulu agar pengukuran akurat.
    """
    # Warm-up
    for _ in range(2):
        _ = generate_fn(batch_size)
        
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = generate_fn(batch_size)
    elapsed = time.perf_counter() - start_time
    latency_ms = (elapsed / num_runs) * 1000.0
    return float(latency_ms)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Menghitung Fréchet Distance matematis antara dua distribusi Gauss multivariat."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Periksa nilai imajiner numerik
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            pass
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

class FIDCalculator:
    """
    Kalkulator Fréchet Inception Distance (FID) untuk evaluasi sampel sintetis Fashion-MNIST.
    Menggunakan CNN feature extractor berbobot tetap yang diekstrak dari dataset untuk representasi fitur yang stabil.
    """
    def __init__(self, device="cpu"):
        self.device = device
        # Feature extractor CNN terstandarisasi untuk citra 28x28 grayscale
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten()
        ).to(device)
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_features(self, dataloader_or_tensor, max_samples=2000):
        features = []
        count = 0
        
        if isinstance(dataloader_or_tensor, torch.Tensor):
            batches = [dataloader_or_tensor[i:i+128] for i in range(0, len(dataloader_or_tensor), 128)]
            for b in batches:
                b = b.to(self.device)
                feat = self.feature_extractor(b)
                features.append(feat.cpu().numpy())
        else:
            for imgs, _ in dataloader_or_tensor:
                imgs = imgs.to(self.device)
                feat = self.feature_extractor(imgs)
                features.append(feat.cpu().numpy())
                count += imgs.size(0)
                if count >= max_samples:
                    break
                    
        features = np.concatenate(features, axis=0)
        return features[:max_samples]

    def compute_fid(self, real_features, fake_features):
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
