"""
Script untuk menghitung ulang FID Score menggunakan InceptionV3 pretrained.
Memuat checkpoint model yang sudah dilatih dan mengevaluasi FID secara akurat.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from src.dataset import get_dataloaders, set_seed
from src.models.autoencoders import (
    VanillaAutoencoder,
    SparseAutoencoder,
    VariationalAutoencoder,
)
from src.models.gan import VanillaGenerator, DCGANGenerator
from src.models.ddpm import DiffusionUNet, GaussianDiffusion
from src.metrics import FIDCalculator

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    set_seed(42)

    # === 1. Load test data ===
    print("\n[1/6] Memuat test dataloader...")
    _, _, test_loader = get_dataloaders(root="./data", batch_size=64, seed=42)

    # === 2. Init FID Calculator (InceptionV3 pretrained) ===
    print("[2/6] Menginisialisasi FID Calculator (InceptionV3 pretrained)...")
    fid_calc = FIDCalculator(device=device)

    # === 3. Ekstrak fitur real test set ===
    print("[3/6] Mengekstrak fitur dari real test set (2048 sampel)...")
    t0 = time.time()
    real_features = fid_calc.get_features(test_loader, max_samples=2048)
    print(f"   >> Real features shape: {real_features.shape} ({time.time()-t0:.1f}s)")

    # === 4. Load models dan generate samples ===
    print("[4/6] Memuat checkpoint dan membangkitkan sampel sintetis...")
    n_samples = 2048

    # --- Vanilla AE ---
    print("\n   [Vanilla AE] Memuat checkpoint dan sampling z ~ N(0,I)...")
    vanilla_ae = VanillaAutoencoder(latent_dim=64).to(device)
    vanilla_ae.load_state_dict(torch.load("./checkpoints/vanilla_ae.pt", map_location=device))
    vanilla_ae.eval()
    with torch.no_grad():
        fake_vanilla_ae = vanilla_ae.decode(torch.randn(n_samples, 64, device=device))

    # --- Sparse AE ---
    print("   [Sparse AE] Memuat checkpoint dan sampling z ~ N(0,I)...")
    sparse_ae = SparseAutoencoder(latent_dim=64).to(device)
    sparse_ae.load_state_dict(torch.load("./checkpoints/sparse_ae.pt", map_location=device))
    sparse_ae.eval()
    with torch.no_grad():
        fake_sparse_ae = sparse_ae.decode(torch.randn(n_samples, 64, device=device))

    # --- VAE ---
    print("\n   [VAE] Memuat checkpoint...")
    vae = VariationalAutoencoder(latent_dim=64).to(device)
    vae.load_state_dict(torch.load("./checkpoints/vae.pt", map_location=device))
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 64, device=device)
        fake_vae = vae.decode(z)
    print(f"   [VAE] Generated {fake_vae.shape[0]} samples, shape: {fake_vae.shape}")

    # --- Vanilla GAN ---
    print("   [Vanilla GAN] Memuat checkpoint...")
    netG_v = VanillaGenerator(latent_dim=64).to(device)
    netG_v.load_state_dict(torch.load("./checkpoints/vanilla_gan_g.pt", map_location=device))
    netG_v.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 64, device=device)
        fake_vgan = netG_v(z)
    print(f"   [Vanilla GAN] Generated {fake_vgan.shape[0]} samples, shape: {fake_vgan.shape}")

    # --- DCGAN ---
    print("   [DCGAN] Memuat checkpoint...")
    netG_dc = DCGANGenerator(latent_dim=100).to(device)
    netG_dc.load_state_dict(torch.load("./checkpoints/dcgan_g.pt", map_location=device))
    netG_dc.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 100, device=device)
        fake_dcgan = netG_dc(z)
    print(f"   [DCGAN] Generated {fake_dcgan.shape[0]} samples, shape: {fake_dcgan.shape}")

    # --- DDPM ---
    print("   [DDPM] Memuat checkpoint dan menjalankan reverse diffusion...")
    ddpm_unet = DiffusionUNet(in_channels=1, time_emb_dim=64).to(device)
    ddpm_unet.load_state_dict(torch.load("./checkpoints/ddpm_unet.pt", map_location=device))
    ddpm = GaussianDiffusion(ddpm_unet, timesteps=200, device=device).to(device)
    ddpm.eval()
    ddpm_unet.eval()

    ddpm_batches = []
    generated = 0
    batch_sz = 64
    while generated < n_samples:
        this_batch = min(batch_sz, n_samples - generated)
        with torch.no_grad():
            samples = ddpm.sample(shape=(this_batch, 1, 28, 28))
        ddpm_batches.append(samples.cpu())
        generated += this_batch
        print(f"   [DDPM] Generated {generated}/{n_samples} samples...")
    fake_ddpm = torch.cat(ddpm_batches, dim=0)
    print(f"   [DDPM] Total generated: {fake_ddpm.shape[0]}")

    # === 5. Hitung FID ===
    print("\n[5/6] Menghitung FID Score untuk setiap model...")

    print("   >> Ekstrak fitur Vanilla AE...")
    t0 = time.time()
    feat_vanilla_ae = fid_calc.get_features(fake_vanilla_ae, max_samples=n_samples)
    fid_vanilla_ae = fid_calc.compute_fid(real_features, feat_vanilla_ae)
    print(f"   Vanilla AE FID: {fid_vanilla_ae:.2f} ({time.time()-t0:.1f}s)")

    print("   >> Ekstrak fitur Sparse AE...")
    t0 = time.time()
    feat_sparse_ae = fid_calc.get_features(fake_sparse_ae, max_samples=n_samples)
    fid_sparse_ae = fid_calc.compute_fid(real_features, feat_sparse_ae)
    print(f"   Sparse AE FID: {fid_sparse_ae:.2f} ({time.time()-t0:.1f}s)")

    print("   >> Ekstrak fitur VAE...")
    t0 = time.time()
    feat_vae = fid_calc.get_features(fake_vae, max_samples=n_samples)
    fid_vae = fid_calc.compute_fid(real_features, feat_vae)
    print(f"   VAE FID: {fid_vae:.2f} ({time.time()-t0:.1f}s)")

    print("   >> Ekstrak fitur Vanilla GAN...")
    t0 = time.time()
    feat_vgan = fid_calc.get_features(fake_vgan, max_samples=n_samples)
    fid_vgan = fid_calc.compute_fid(real_features, feat_vgan)
    print(f"   Vanilla GAN FID: {fid_vgan:.2f} ({time.time()-t0:.1f}s)")

    print("   >> Ekstrak fitur DCGAN...")
    t0 = time.time()
    feat_dcgan = fid_calc.get_features(fake_dcgan, max_samples=n_samples)
    fid_dcgan = fid_calc.compute_fid(real_features, feat_dcgan)
    print(f"   DCGAN FID: {fid_dcgan:.2f} ({time.time()-t0:.1f}s)")

    print("   >> Ekstrak fitur DDPM...")
    t0 = time.time()
    feat_ddpm = fid_calc.get_features(fake_ddpm, max_samples=n_samples)
    fid_ddpm = fid_calc.compute_fid(real_features, feat_ddpm)
    print(f"   DDPM FID: {fid_ddpm:.2f} ({time.time()-t0:.1f}s)")

    # === 6. Update metrics JSON ===
    print("\n[6/6] Menyimpan hasil FID yang diperbarui...")
    metrics_path = "./outputs/all_models_metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    metrics["vanilla_ae"]["fid_score"] = round(fid_vanilla_ae, 2)
    metrics["sparse_ae"]["fid_score"] = round(fid_sparse_ae, 2)
    metrics["vae"]["fid_score"] = round(fid_vae, 2)
    metrics["vanilla_gan"]["fid_score"] = round(fid_vgan, 2)
    metrics["dcgan"]["fid_score"] = round(fid_dcgan, 2)
    metrics["ddpm"]["fid_score"] = round(fid_ddpm, 2)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "=" * 60)
    print("HASIL FID SCORE (InceptionV3 Pretrained)")
    print("=" * 60)
    print(f"  Vanilla AE  : {fid_vanilla_ae:.2f} (prior N(0,I), diagnostik)")
    print(f"  Sparse AE   : {fid_sparse_ae:.2f} (prior N(0,I), diagnostik)")
    print(f"  VAE         : {fid_vae:.2f}")
    print(f"  Vanilla GAN : {fid_vgan:.2f}")
    print(f"  DCGAN       : {fid_dcgan:.2f}")
    print(f"  DDPM        : {fid_ddpm:.2f}")
    print("=" * 60)
    print(f">> Metrics tersimpan di {metrics_path}")

if __name__ == "__main__":
    main()
