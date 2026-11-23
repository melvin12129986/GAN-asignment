import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.dataset import get_dataloaders, set_seed
from src.models.mlp_classifier import MLPClassifier
from src.models.autoencoders import FrozenLatentDecoder, VanillaAutoencoder
from src.metrics import compute_mse, compute_psnr, compute_ssim

def run_mlp_limitation_experiments(device=None, epochs_classifier=15, epochs_decoder=15):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"BAGIAN 2: PEMBUKTIAN BATASAN TEORETIS & EMPIRIS MLP CLASSIFIER (Device: {device})")
    print("=" * 70)
    
    set_seed(42)
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./outputs", exist_ok=True)
    
    train_loader, val_loader, test_loader = get_dataloaders(root="./data", batch_size=128, seed=42)
    
    # -------------------------------------------------------------
    # TAHAP 1: Pelatihan MLP Classifier Berjenjang (784 -> 256 -> 64 -> 10)
    # -------------------------------------------------------------
    print("\n[Tahap 1] Melatih MLP Classifier hingga akurasi test set >= 88%...")
    classifier = MLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    for epoch in range(1, epochs_classifier + 1):
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = classifier(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
        train_acc = correct / total
        
        # Evaluasi pada Validation Set
        classifier.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = classifier(x)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / val_total
        print(f"Epoch [{epoch:02d}/{epochs_classifier:02d}] Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
    
    # Evaluasi Final pada Test Set
    classifier.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = classifier(x)
            test_correct += (logits.argmax(dim=1) == y).sum().item()
            test_total += x.size(0)
    final_test_acc = test_correct / test_total
    print(f">> Akurasi Test Set MLP Classifier: {final_test_acc * 100:.2f}% (Target: >= 88.0%)")
    assert final_test_acc >= 0.88, f"Akurasi belum mencapai 88%, dapat: {final_test_acc*100:.2f}%"
    
    # Simpan checkpoint bobot
    torch.save(classifier.state_dict(), "./checkpoints/mlp_classifier.pt")
    
    # Ekstraksi bobot W1, W2, W3
    W1 = classifier.fc1.weight.detach().cpu()  # [256, 784]
    W2 = classifier.fc2.weight.detach().cpu()  # [64, 256]
    W3 = classifier.fc3.weight.detach().cpu()  # [10, 64]
    
    # -------------------------------------------------------------
    # TAHAP 2: Proyeksi Balik Bobot Langsung (Top-Down Weight Transposition)
    # y_c in R^{10} -> z_hat = ReLU(W3^T y_c) -> h1_hat = ReLU(W2^T z_hat) -> x_hat = Sigmoid(W1^T h1_hat)
    # -------------------------------------------------------------
    print("\n[Tahap 2] Proyeksi Balik Bobot Langsung dari Vektor Kelas (Top-Down Weight Transposition)...")
    recon_top_down = []
    for c in range(10):
        y_c = torch.zeros(10)
        y_c[c] = 1.0
        
        # Operasi transpose matrix
        z_hat = torch.relu(torch.matmul(W3.t(), y_c))
        h1_hat = torch.relu(torch.matmul(W2.t(), z_hat))
        x_hat = torch.sigmoid(torch.matmul(W1.t(), h1_hat))
        recon_top_down.append(x_hat.view(1, 28, 28))
    recon_top_down = torch.stack(recon_top_down)
    print(f">> Berhasil membangkitkan 10 citra proyeksi W^T y_c (Shape: {recon_top_down.shape})")
    
    # -------------------------------------------------------------
    # TAHAP 3: Proyeksi Balik dari Latent Features Citra Uji (Latent Transpose Projection)
    # x_hat = Sigmoid(W1^T ReLU(W2^T z_MLP))
    # -------------------------------------------------------------
    print("\n[Tahap 3] Proyeksi Balik dari Latent Features Citra Uji (Latent Transpose Projection)...")
    # Ambil 1 batch test sample
    test_iter = iter(test_loader)
    sample_imgs, sample_labels = next(test_iter)
    sample_imgs = sample_imgs.to(device)
    
    with torch.no_grad():
        z_mlp = classifier.extract_features(sample_imgs)
        
    z_mlp_cpu = z_mlp.cpu()
    # x_hat = Sigmoid(W1^T ReLU(W2^T z_MLP))
    h1_hat_latent = torch.relu(torch.matmul(z_mlp_cpu, W2))  # z_mlp [B, 64] * W2^T [64, 256] -> [B, 256]
    x_hat_latent = torch.sigmoid(torch.matmul(h1_hat_latent, W1)).view(-1, 1, 28, 28)
    
    mse_t3 = compute_mse(sample_imgs.cpu(), x_hat_latent)
    psnr_t3 = compute_psnr(sample_imgs.cpu(), x_hat_latent)
    ssim_t3 = compute_ssim(sample_imgs.cpu(), x_hat_latent)
    print(f">> Proyeksi Transpos Bobot Laten: MSE={mse_t3:.4f}, PSNR={psnr_t3:.2f} dB, SSIM={ssim_t3:.4f}")
    print("   (Membuktikan bobot maju klasifikasi tidak dapat dibalik menjadi rekonstruksi yang koheren!)")
    
    # -------------------------------------------------------------
    # TAHAP 4: Evaluasi Kandungan Informasi Latent Features (Frozen Latent Decoding)
    # Bekukan MLP Classifier (requires_grad = False). Latih Decoder terpisah (64 -> 256 -> 784)
    # -------------------------------------------------------------
    print("\n[Tahap 4] Melatih Frozen Latent Decoder (64 -> 256 -> 784) pada z_MLP yang dibekukan...")
    for p in classifier.parameters():
        p.requires_grad = False
    classifier.eval()
    
    frozen_decoder = FrozenLatentDecoder().to(device)
    dec_optimizer = optim.Adam(frozen_decoder.parameters(), lr=1e-3)
    dec_criterion = nn.MSELoss()
    
    for epoch in range(1, epochs_decoder + 1):
        frozen_decoder.train()
        dec_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z_frozen = classifier.extract_features(x)
            dec_optimizer.zero_grad()
            x_recon = frozen_decoder(z_frozen)
            loss = dec_criterion(x_recon, x)
            loss.backward()
            dec_optimizer.step()
            dec_loss += loss.item() * x.size(0)
        dec_loss /= len(train_loader.dataset)
        if epoch % 2 == 0 or epoch == epochs_decoder:
            print(f"Decoder Epoch [{epoch:02d}/{epochs_decoder:02d}] Recon Loss (MSE): {dec_loss:.4f}")
            
    torch.save(frozen_decoder.state_dict(), "./checkpoints/frozen_latent_decoder.pt")
    
    # Evaluasi metrik rekonstruksi Frozen Latent Decoder pada Test Set
    frozen_decoder.eval()
    all_recons = []
    all_reals = []
    all_z_test = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            z_f = classifier.extract_features(x)
            xr = frozen_decoder(z_f)
            all_recons.append(xr.cpu())
            all_reals.append(x.cpu())
            all_z_test.append(z_f.cpu())
            
    all_recons = torch.cat(all_recons, dim=0)
    all_reals = torch.cat(all_reals, dim=0)
    all_z_test = torch.cat(all_z_test, dim=0)
    
    fld_mse = compute_mse(all_reals, all_recons)
    fld_psnr = compute_psnr(all_reals, all_recons)
    fld_ssim = compute_ssim(all_reals[:1000], all_recons[:1000])  # subset 1000 untuk efisiensi
    print(f">> Metrik Frozen Latent Decoder: MSE={fld_mse:.4f}, PSNR={fld_psnr:.2f} dB, SSIM={fld_ssim:.4f}")
    
    # -------------------------------------------------------------
    # TAHAP 5: Uji Sampling Generatif pada Ruang Laten MLP Classifier
    # z_rand ~ N(mu_z, diag(sigma_z^2)) -> Decoder Tahap 4
    # -------------------------------------------------------------
    print("\n[Tahap 5] Uji Sampling Generatif pada Ruang Laten MLP Classifier...")
    mu_z = torch.mean(all_z_test, dim=0)
    sigma_z = torch.std(all_z_test, dim=0) + 1e-6
    
    num_samples = 16
    z_rand = torch.randn(num_samples, 64) * sigma_z + mu_z
    z_rand = z_rand.to(device)
    
    with torch.no_grad():
        mlp_generative_samples = frozen_decoder(z_rand).cpu()
    print(f">> Berhasil menguji prior sampling pada z_MLP (Shape: {mlp_generative_samples.shape})")
    print("   (Hasil visual menghasilkan artefak tak bermakna karena tiadanya regularisasi continuous prior!)")
    
    results = {
        "final_test_acc": final_test_acc,
        "W1": W1, "W2": W2, "W3": W3,
        "recon_top_down": recon_top_down,
        "sample_imgs": sample_imgs.cpu()[:16],
        "x_hat_latent": x_hat_latent[:16],
        "frozen_decoder_recons": all_recons[:16],
        "fld_metrics": {"mse": fld_mse, "psnr": fld_psnr, "ssim": fld_ssim},
        "mlp_generative_samples": mlp_generative_samples,
        "mu_z": mu_z, "sigma_z": sigma_z
    }
    torch.save(results, "./outputs/mlp_experiments_results.pt")
    print("=" * 70)
    print("BAGIAN 2 SELESAI & SELURUH HASIL DISIMPAN KE ./outputs/mlp_experiments_results.pt")
    print("=" * 70)
    return results

if __name__ == "__main__":
    run_mlp_limitation_experiments()
