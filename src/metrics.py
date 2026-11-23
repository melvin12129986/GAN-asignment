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
    if real.shape != recon.shape:
        raise ValueError(
            f"real_imgs dan recon_imgs harus memiliki shape yang sama, "
            f"didapat {real.shape} dan {recon.shape}"
        )
    if real.ndim != 4 or real.shape[1] != 1:
        raise ValueError(
            f"Citra harus bertipe NCHW grayscale dengan satu channel, didapat shape {real.shape}"
        )
    
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
    def synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Warm-up
    for _ in range(2):
        _ = generate_fn(batch_size)
        synchronize()

    synchronize()
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = generate_fn(batch_size)
        synchronize()
    elapsed = time.perf_counter() - start_time
    latency_ms = (elapsed / num_runs) * 1000.0
    return float(latency_ms)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Menghitung Fréchet Distance matematis antara dua distribusi Gauss multivariat."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError("Dimensi mean dari kedua distribusi harus sama.")
    if sigma1.shape != sigma2.shape or sigma1.shape[0] != mu1.shape[0]:
        raise ValueError("Dimensi covariance harus sesuai dengan dimensi mean.")

    sigma1 = (sigma1 + sigma1.T) * 0.5
    sigma2 = (sigma2 + sigma2.T) * 0.5
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    sqrt_sigma1 = linalg.sqrtm(sigma1)
    if np.iscomplexobj(sqrt_sigma1):
        if not np.allclose(sqrt_sigma1.imag, 0, atol=1e-3):
            return float("inf")
        sqrt_sigma1 = sqrt_sigma1.real

    covmean = linalg.sqrtm(sqrt_sigma1.dot(sigma2).dot(sqrt_sigma1))
    if not np.isfinite(covmean).all():
        return float("inf")

    if np.iscomplexobj(covmean):
        if not np.allclose(covmean.imag, 0, atol=1e-3):
            return float("inf")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    distance = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(max(distance, 0.0))

class FIDCalculator:
    """
    Kalkulator Fréchet Inception Distance (FID) menggunakan InceptionV3 pretrained (ImageNet).
    Standar industri: mengekstrak fitur dari pool3 layer (2048-dim) InceptionV3.
    Citra Fashion-MNIST 28x28 grayscale di-resize ke 299x299 dan dikonversi ke 3-channel RGB.
    """
    def __init__(self, device="cpu"):
        import torchvision.models as models
        import torchvision.transforms as T

        self.device = device
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.fc = torch.nn.Identity()
        inception.eval()
        inception.to(device)
        for p in inception.parameters():
            p.requires_grad = False
        self.inception = inception

        self.preprocess = T.Compose([
            T.Resize((299, 299), antialias=True),
            T.Lambda(
                lambda x: x.expand(-1, 3, -1, -1)
                if x.shape[1] == 1 else x
            ),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_features(self, dataloader_or_tensor, max_samples=2048):
        """Ekstrak fitur 2048-dim dari InceptionV3 pool3 layer."""
        if max_samples <= 0:
            raise ValueError("max_samples harus lebih besar dari 0.")

        features = []
        count = 0

        if isinstance(dataloader_or_tensor, torch.Tensor):
            for start in range(0, len(dataloader_or_tensor), 32):
                b = dataloader_or_tensor[start:start + 32]
                b = b.to(self.device)
                b = self.preprocess(b)
                feat = self.inception(b)
                features.append(feat.cpu().numpy())
                count += b.size(0)
                if count >= max_samples:
                    break
        else:
            for imgs, _ in dataloader_or_tensor:
                imgs = imgs.to(self.device)
                imgs = self.preprocess(imgs)
                feat = self.inception(imgs)
                features.append(feat.cpu().numpy())
                count += imgs.size(0)
                if count >= max_samples:
                    break

        features = np.concatenate(features, axis=0)
        return features[:max_samples]

    def compute_fid(self, real_features, fake_features):
        if real_features.ndim != 2 or fake_features.ndim != 2:
            raise ValueError("Feature arrays harus berupa matriks 2D.")
        if real_features.shape[1] != fake_features.shape[1]:
            raise ValueError("Real dan fake features harus memiliki dimensi yang sama.")
        if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
            raise ValueError("Untuk FID diperlukan minimal dua sampel per distribusi.")

        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
