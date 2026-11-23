import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.autoencoders import VanillaAutoencoder, SparseAutoencoder, VariationalAutoencoder, FrozenLatentDecoder
from src.models.gan import VanillaGenerator, DCGANGenerator
from src.models.mlp_classifier import MLPClassifier

# Set font dan style profesional untuk publikasi ilmiah
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def generate_all_figures(output_dir="./outputs/figures", device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(f"MEMBUAT 5 FIGUR KOMPARATIF WAJIB (Format 300 DPI Publikasi)")
    print("=" * 70)
    
    # Load aset evaluasi dan hasil eksperimen ke CPU untuk rendering matplotlib
    eval_assets = torch.load("./outputs/evaluation_samples.pt", map_location="cpu")
    mlp_res = torch.load("./outputs/mlp_experiments_results.pt", map_location="cpu")
    training_curves = torch.load("./outputs/training_curves.pt", map_location="cpu")
    with open("./outputs/all_models_metrics.json", "r") as f:
        metrics = json.load(f)
    
    # -------------------------------------------------------------------------
    # FIGUR 1: Grid Rekonstruksi Side-by-Side
    # Baris 1: Ground Truth
    # Baris 2: Vanilla AE
    # Baris 3: Sparse AE
    # Baris 4: VAE
    # Baris 5: Frozen Latent Decoder (MLP)
    # -------------------------------------------------------------------------
    print("[1/5] Membuat Figur 1: Grid Rekonstruksi Side-by-Side...")
    num_samples = 10
    fig, axes = plt.subplots(5, num_samples, figsize=(14, 7), dpi=300)
    
    row_titles = [
        "Ground Truth",
        "Vanilla AE (z=64)",
        "Sparse AE (z=64, L1)",
        "VAE (z=64, Prob.)",
        "MLP Latent Dec. (z=64)"
    ]
    
    imgs_gt = eval_assets["fixed_test_imgs"][:num_samples].cpu()
    imgs_vae = eval_assets["rec_vanilla_ae"][:num_samples].cpu()
    imgs_sae = eval_assets["rec_sparse_ae"][:num_samples].cpu()
    imgs_prob_vae = eval_assets["rec_vae"][:num_samples].cpu()
    imgs_mlp_dec = mlp_res["frozen_decoder_recons"][:num_samples].cpu()
    
    rows_data = [imgs_gt, imgs_vae, imgs_sae, imgs_prob_vae, imgs_mlp_dec]
    
    for row_idx, data in enumerate(rows_data):
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx]
            img = data[col_idx, 0].numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col_idx == 0:
                ax.text(-0.25, 0.5, row_titles[row_idx], transform=ax.transAxes,
                        fontsize=10, fontweight="bold", va="center", ha="right")
                
    plt.suptitle("Figur 1: Komparasi Visual Rekonstruksi Citra Side-by-Side pada Test Set Fashion-MNIST",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.12, 0.02, 1, 0.95])
    fig1_path = os.path.join(output_dir, "figur1_rekonstruksi_side_by_side.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Figur 1 disimpan di {fig1_path}")

    # -------------------------------------------------------------------------
    # FIGUR 2: Grid Random Generative Sampling (4x4 per Panel)
    # Panel A: VAE, Panel B: Vanilla GAN, Panel C: DCGAN, Panel D: DDPM
    # -------------------------------------------------------------------------
    print("[2/5] Membuat Figur 2: Grid Random Generative Sampling (4 Panels)...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)
    
    models_gen = [
        ("vae", "Panel A: VAE (Prior z ~ N(0,I))", eval_assets["rand_vae"][:16].cpu()),
        ("vanilla_gan", "Panel B: Vanilla GAN (MLP)", eval_assets["rand_vgan"][:16].cpu()),
        ("dcgan", "Panel C: DCGAN (Conv)", eval_assets["rand_dcgan"][:16].cpu()),
        ("ddpm", "Panel D: DDPM (Diffusion x_T->x_0)", eval_assets["rand_ddpm"][:16].cpu())
    ]
    
    for i, (model_key, title, samples) in enumerate(models_gen):
        grid = make_grid(samples, nrow=4, padding=2, normalize=False)
        axes[i].imshow(grid[0].numpy(), cmap="gray", vmin=0, vmax=1)
        fid_score = metrics[model_key]["fid_score"]
        fid_label = (
            f"FID: {fid_score:.2f}"
            if isinstance(fid_score, (int, float))
            else f"FID: {fid_score}"
        )
        axes[i].set_title(f"{title}\n{fid_label}", fontsize=10, fontweight="bold", pad=8)
        axes[i].axis("off")
        
    plt.suptitle("Figur 2: Sintesis Citra Generatif Acak (Grid 4x4) Antarmodel Generatif pada Fashion-MNIST",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2_path = os.path.join(output_dir, "figur2_random_generative_sampling.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Figur 2 disimpan di {fig2_path}")

    # -------------------------------------------------------------------------
    # FIGUR 3: Bukti Empiris Batasan Bobot dan Latent Features MLP Classifier
    # 4 Panel:
    # (1) Citra Asli (Ground Truth)
    # (2) Proyeksi Balik Transpos Bobot W^T dari y_c dan z_MLP
    # (3) Rekonstruksi Frozen Latent Decoder vs Vanilla AE
    # (4) Random Prior Sampling z_rand ~ N(mu_z, diag(sigma_z^2))
    # -------------------------------------------------------------------------
    print("[3/5] Membuat Figur 3: Bukti Empiris Batasan Bobot dan Latent Features MLP Classifier...")
    fig = plt.figure(figsize=(15, 8), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    # Subplot 1: Ground Truth Citra Uji (8 Sampel)
    ax1 = fig.add_subplot(gs[0, 0])
    gt_grid = make_grid(mlp_res["sample_imgs"][:8].cpu(), nrow=8, padding=2, normalize=False)
    ax1.imshow(gt_grid[0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Panel (1): Citra Asli Uji (Ground Truth Real Citra)", fontsize=10, fontweight="bold")
    ax1.axis("off")
    
    # Subplot 2: Proyeksi Transpos W^T y_c (10 Kelas) & W^T z_MLP
    ax2 = fig.add_subplot(gs[0, 1])
    # Gabungkan proyeksi kelas y_c (baris atas) dan proyeksi laten z_MLP (baris bawah)
    top_down_yc = mlp_res["recon_top_down"][:8].cpu()
    proj_latent = mlp_res["x_hat_latent"][:8].cpu()
    stacked_proj = torch.cat([top_down_yc, proj_latent], dim=0)
    proj_grid = make_grid(stacked_proj, nrow=8, padding=2, normalize=False)
    ax2.imshow(proj_grid[0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Panel (2): Proyeksi Balik Transpos Bobot W^T (W^T y_c & W^T z_MLP)", fontsize=10, fontweight="bold")
    ax2.axis("off")
    
    # Subplot 3: Frozen Latent Decoder vs Vanilla AE
    ax3 = fig.add_subplot(gs[1, 0])
    rec_mlp = mlp_res["frozen_decoder_recons"][:8].cpu()
    rec_vae = eval_assets["rec_vanilla_ae"][:8].cpu()
    comp_recon = torch.cat([rec_mlp, rec_vae], dim=0)
    recon_grid = make_grid(comp_recon, nrow=8, padding=2, normalize=False)
    ax3.imshow(recon_grid[0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax3.set_title("Panel (3): Rekonstruksi Frozen Latent Decoder (Atas) vs Vanilla AE (Bawah)", fontsize=10, fontweight="bold")
    ax3.axis("off")
    
    # Subplot 4: Prior Sampling z_rand ~ N(mu, sigma^2) pada Laten MLP
    ax4 = fig.add_subplot(gs[1, 1])
    sample_grid = make_grid(mlp_res["mlp_generative_samples"][:16].cpu(), nrow=8, padding=2, normalize=False)
    ax4.imshow(sample_grid[0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax4.set_title("Panel (4): Random Prior Sampling pada Ruang Laten MLP (z ~ N(mu_z, sigma_z^2))", fontsize=10, fontweight="bold")
    ax4.axis("off")
    
    plt.suptitle("Figur 3: Bukti Empiris Batasan Bobot dan Latent Features MLP Classifier untuk Image Generation",
                 fontsize=12, fontweight="bold", y=0.98)
    fig3_path = os.path.join(output_dir, "figur3_bukti_batasan_mlp_classifier.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Figur 3 disimpan di {fig3_path}")

    # -------------------------------------------------------------------------
    # FIGUR 4: Linear Latent Space Interpolation
    # z(alpha) = (1-alpha)*zA + alpha*zB, alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN
    # -------------------------------------------------------------------------
    print("[4/5] Membuat Figur 4: Linear Latent Space Interpolation...")
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Muat checkpoint model untuk interpolasi laten
    vae_model = VariationalAutoencoder(latent_dim=64).to(device)
    vae_model.load_state_dict(torch.load("./checkpoints/vae.pt", map_location=device))
    vae_model.eval()
    
    vanilla_ae = VanillaAutoencoder(latent_dim=64).to(device)
    vanilla_ae.load_state_dict(torch.load("./checkpoints/vanilla_ae.pt", map_location=device))
    vanilla_ae.eval()
    
    sparse_ae = SparseAutoencoder(latent_dim=64).to(device)
    sparse_ae.load_state_dict(torch.load("./checkpoints/sparse_ae.pt", map_location=device))
    sparse_ae.eval()
    
    vgan_g = VanillaGenerator(latent_dim=64).to(device)
    vgan_g.load_state_dict(torch.load("./checkpoints/vanilla_gan_g.pt", map_location=device))
    vgan_g.eval()
    
    dcgan_g = DCGANGenerator(latent_dim=100).to(device)
    dcgan_g.load_state_dict(torch.load("./checkpoints/dcgan_g.pt", map_location=device))
    dcgan_g.eval()
    
    # Ambil 2 citra uji kelas kontras (misal T-shirt index 0 dan Ankle Boot index 9)
    test_imgs = eval_assets["fixed_test_imgs"].to(device)
    test_labels = eval_assets["fixed_labels"]
    
    idx_a, idx_b = 0, 1
    for i in range(len(test_labels)):
        for j in range(i+1, len(test_labels)):
            if test_labels[i].item() != test_labels[j].item():
                idx_a, idx_b = i, j
                break
        if idx_a != idx_b:
            break
            
    img_a = test_imgs[idx_a:idx_a+1]
    img_b = test_imgs[idx_b:idx_b+1]
    
    with torch.no_grad():
        # Latent AE
        zA_vae = vanilla_ae.encode(img_a)
        zB_vae = vanilla_ae.encode(img_b)
        
        zA_sae = sparse_ae.encode(img_a)
        zB_sae = sparse_ae.encode(img_b)
        
        mu_a, _ = vae_model.encode(img_a)
        mu_b, _ = vae_model.encode(img_b)
        
        # Latent GAN (Prior Space)
        torch.manual_seed(101)
        zA_vgan = torch.randn(1, 64, device=device)
        zB_vgan = torch.randn(1, 64, device=device)
        
        zA_dcgan = torch.randn(1, 100, device=device)
        zB_dcgan = torch.randn(1, 100, device=device)
        
        interp_results = {
            "Vanilla AE": [],
            "Sparse AE": [],
            "VAE": [],
            "Vanilla GAN": [],
            "DCGAN": []
        }
        
        for alpha in alphas:
            # Interpolasi linier
            z_v = (1.0 - alpha) * zA_vae + alpha * zB_vae
            interp_results["Vanilla AE"].append(vanilla_ae.decode(z_v).cpu()[0, 0].numpy())
            
            z_s = (1.0 - alpha) * zA_sae + alpha * zB_sae
            interp_results["Sparse AE"].append(sparse_ae.decode(z_s).cpu()[0, 0].numpy())
            
            z_prob = (1.0 - alpha) * mu_a + alpha * mu_b
            interp_results["VAE"].append(vae_model.decode(z_prob).cpu()[0, 0].numpy())
            
            z_vg = (1.0 - alpha) * zA_vgan + alpha * zB_vgan
            interp_results["Vanilla GAN"].append(vgan_g(z_vg).cpu()[0, 0].numpy())
            
            z_dc = (1.0 - alpha) * zA_dcgan + alpha * zB_dcgan
            interp_results["DCGAN"].append(dcgan_g(z_dc).cpu()[0, 0].numpy())

    fig, axes = plt.subplots(5, len(alphas), figsize=(12, 7.5), dpi=300)
    model_names = ["Vanilla AE", "Sparse AE", "VAE", "Vanilla GAN", "DCGAN"]
    
    for row_idx, m_name in enumerate(model_names):
        for col_idx, alpha in enumerate(alphas):
            ax = axes[row_idx, col_idx]
            img = interp_results[m_name][col_idx]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"α = {alpha:.1f}", fontsize=10, fontweight="bold", pad=5)
            if col_idx == 0:
                ax.text(-0.2, 0.5, m_name, transform=ax.transAxes,
                        fontsize=10, fontweight="bold", va="center", ha="right")
                
    plt.suptitle("Figur 4: Linear Latent Space Interpolation z(α) = (1-α)z_A + α z_B antarmodel",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.12, 0.02, 1, 0.95])
    fig4_path = os.path.join(output_dir, "figur4_linear_latent_interpolation.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Figur 4 disimpan di {fig4_path}")

    # -------------------------------------------------------------------------
    # FIGUR 5: Training Loss Curves dan DDPM Denoising Trajectory
    # Panel Atas: Kurva loss 6 model
    # Panel Bawah: Reverse diffusion trajectory (x_T -> x_0)
    # -------------------------------------------------------------------------
    print("[5/5] Membuat Figur 5: Training Loss Curves & DDPM Denoising Trajectory...")
    fig = plt.figure(figsize=(15, 8.5), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.8], hspace=0.35, wspace=0.25)
    
    # 1. Loss Autoencoders
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(training_curves["vanilla_ae"], label="Vanilla AE (Recon MSE)", color="#1f77b4", lw=2)
    ax1.plot(training_curves["sparse_ae"], label="Sparse AE (MSE + L1)", color="#2ca02c", lw=2)
    ax1.plot(training_curves["vae"], label="VAE (Recon + KL)", color="#ff7f0e", lw=2)
    ax1.set_title("Kurva Loss: Autoencoders & VAE", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=9)
    ax1.set_ylabel("Loss", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8)
    
    # 2. Loss GANs (Vanilla vs DCGAN)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(training_curves["vanilla_gan"]["g_loss"], label="Vanilla GAN (G)", color="#d62728", lw=1.5, ls="--")
    ax2.plot(training_curves["vanilla_gan"]["d_loss"], label="Vanilla GAN (D)", color="#9467bd", lw=1.5, ls="--")
    ax2.plot(training_curves["dcgan"]["g_loss"], label="DCGAN (G)", color="#d62728", lw=2)
    ax2.plot(training_curves["dcgan"]["d_loss"], label="DCGAN (D)", color="#9467bd", lw=2)
    ax2.set_title("Kurva Loss: Vanilla GAN vs DCGAN", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=9)
    ax2.set_ylabel("Adversarial BCE Loss", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)
    
    # 3. Loss DDPM
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(training_curves["ddpm"], label="DDPM Noise Pred MSE", color="#8c564b", lw=2)
    ax3.set_title("Kurva Loss: DDPM (Supervised Noise Prediction)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=9)
    ax3.set_ylabel("MSE Loss", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=8)
    
    # Panel Bawah: DDPM Reverse Diffusion Trajectory x_T -> x_0 (Langkah demi langkah)
    ax_traj = fig.add_subplot(gs[1, :])
    traj_imgs = eval_assets["ddpm_trajectory"]  # list of [B, 1, 28, 28]
    # Ambil sampel pertama sepanjang lintasan
    step_imgs = [t[0, 0].numpy() for t in traj_imgs]
    num_steps = len(step_imgs)
    
    # Buat multi-axis di dalam ax_traj untuk menampilkan derau -> citra
    ax_traj.axis("off")
    sub_axes = [fig.add_subplot(gs[1, :], frameon=False)]
    
    # Tampilkan deretan lintasan
    traj_grid_imgs = torch.stack([t[0] for t in traj_imgs], dim=0)
    traj_grid = make_grid(traj_grid_imgs, nrow=num_steps, padding=4, normalize=False)
    ax_traj.imshow(traj_grid[0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax_traj.set_title("Sekuens Reverse Denoising DDPM Langkah Demi Langkah: x_T (Pure Noise) ───> x_0 (Synthesized Clean Image)",
                      fontsize=11, fontweight="bold", pad=8)
    
    plt.suptitle("Figur 5: Dinamika Kurva Pelatihan Enam Model dan Lintasan Denoising Reverse DDPM",
                 fontsize=13, fontweight="bold", y=0.98)
    fig5_path = os.path.join(output_dir, "figur5_training_loss_curves_dan_ddpm_trajectory.png")
    plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Figur 5 disimpan di {fig5_path}")
    print("=" * 70)
    print("SELURUH 5 FIGUR VISUALISASI BERHASIL DIBUAT DENGAN RESOLUSI 300 DPI!")
    print("=" * 70)

if __name__ == "__main__":
    generate_all_figures()
