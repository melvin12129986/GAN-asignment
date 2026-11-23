# Studi Komparatif Empiris Model Generatif & Pembuktian Batasan Teoretis Model Klasifikasi untuk Image Generation

**Mata Kuliah:** Deep Learning (D6421) — Sesi 07 (Generative Models terintegrasi Sesi 06 Autoencoders)  
**Tugas:** Take-Home Mandiri / Individu (Proyek Eksperimen Komparatif)  
**Dataset Benchmark:** Fashion-MNIST (70.000 citra, resolusi 28x28 grayscale, 10 kategori pakaian)  
**Standardisasi Preprocessing:** Split 80% Train (56.000 citra), 10% Validation (7.000 citra), 10% Test (7.000 citra) dengan fixed random seed (`seed=42`).

---

## 1. Ringkasan Eksekutif & Struktur Repositori

Repositori ini mengimplementasikan, melatih, mengevaluasi, serta membandingkan secara empiris enam arsitektur model generatif modern dan satu arsitektur Multi-Layer Perceptron (MLP) Classifier pendamping:
1. **Vanilla Autoencoder** (Deterministik MLP, $z \in \mathbb{R}^{64}$)
2. **Sparse Autoencoder** (Penalti sparsity L1 $\lambda \|z\|_1$, $z \in \mathbb{R}^{64}$)
3. **Variational Autoencoder (VAE)** (Pemodelan probabilistik, reparameterization trick $z \sim q_\phi(z|x)$)
4. **Vanilla Generative Adversarial Network (GAN)** (Generator & Discriminator MLP, prior derau acak $z \in \mathbb{R}^{64}$)
5. **Deep Convolutional GAN (DCGAN)** (Generator ConvTranspose2d & Discriminator Conv2d, prior $z \in \mathbb{R}^{100}$)
6. **Denoising Diffusion Probabilistic Model (DDPM)** (U-Net ringan dengan sinusoidal time embedding, reverse diffusion $x_T \to x_0$, $T=200$)
7. **MLP Classifier Pendamping** (784 $\to$ 256 $\to$ 64 $\to$ 10, Cross-Entropy Loss, akurasi test set $\ge 88\%$)

### Struktur Direktori
```text
DL GAN/
├── checkpoints/              # Checkpoint bobot terlatih PyTorch (.pt)
│   ├── mlp_classifier.pt
│   ├── frozen_latent_decoder.pt
│   ├── vanilla_ae.pt
│   ├── sparse_ae.pt
│   ├── vae.pt
│   ├── vanilla_gan_g.pt
│   ├── vanilla_gan_d.pt
│   ├── dcgan_g.pt
│   ├── dcgan_d.pt
│   └── ddpm_unet.pt
├── data/                     # Dataset Fashion-MNIST (diunduh legal via torchvision)
├── outputs/                  # Hasil evaluasi numerik dan kurva loss
│   ├── all_models_metrics.json
│   ├── mlp_experiments_results.pt
│   ├── training_curves.pt
│   ├── evaluation_samples.pt
│   └── figures/              # 5 Figur visualisasi komparatif wajib (300 DPI)
│       ├── figur1_rekonstruksi_side_by_side.png
│       ├── figur2_random_generative_sampling.png
│       ├── figur3_bukti_batasan_mlp_classifier.png
│       ├── figur4_linear_latent_interpolation.png
│       └── figur5_training_loss_curves_dan_ddpm_trajectory.png
├── src/                      # Kode modular PyTorch
│   ├── models/
│   │   ├── autoencoders.py   # Vanilla AE, Sparse AE, VAE, Frozen Latent Decoder
│   │   ├── gan.py            # Vanilla GAN & DCGAN (Generator & Discriminator)
│   │   ├── ddpm.py           # UNet dengan time embedding & Gaussian Diffusion
│   │   └── mlp_classifier.py # MLP Classifier berjenjang (784 -> 256 -> 64 -> 10)
│   ├── dataset.py            # Data pipeline terstandarisasi (80/10/10 split, seed=42)
│   ├── metrics.py            # MSE, PSNR, SSIM, Latency batch 64, FID Score
│   ├── experiments_mlp_limits.py # Tahap 1-5 pembuktian batasan empiris MLP Classifier
│   ├── train_generative_models.py # Pipeline pelatihan 6 model generatif & logging
│   └── generate_figures.py   # Generator 5 figur komparatif publikasi ilmiah (300 DPI)
├── main.py                   # Runner CLI utama
├── requirements.txt          # Daftar dependensi pustaka
└── README.md                 # Dokumentasi komprehensif proyek
```

---

## 2. Petunjuk Instalasi & Eksekusi

### Kebutuhan Sistem
- Python 3.10+ (Diuji pada Python 3.12 64-bit)
- PyTorch dengan dukungan CUDA (Akselerasi GPU NVIDIA RTX 3060) atau CPU

### Langkah Instalasi
```bash
# Clone atau buka direktori proyek
cd "DL GAN"

# Instalasi dependensi
pip install -r requirements.txt
```

### Cara Menjalankan
Seluruh alur eksperimen dapat dijalankan melalui berkas runner `main.py`:

```bash
# Menjalankan seluruh alur eksperimen dari awal hingga akhir (termasuk 5 figur):
python main.py --mode all

# Menjalankan Bagian 2 saja (Pembuktian Batasan MLP Classifier Tahap 1-5):
python main.py --mode mlp_limits

# Menjalankan Pelatihan & Evaluasi 6 Model Generatif:
python main.py --mode train_generative

# Menghasilkan ulang 5 Figur Visualisasi Wajib (300 DPI):
python main.py --mode figures
```

---

## 3. Hasil Pengujian Kuantitatif (Template Tabel Metrik Bagian IV)

Seluruh metrik dievaluasi pada **Test Set (7.000 citra)** yang belum pernah dilihat oleh model saat pelatihan:

| Arsitektur Model | Dim Laten | Training Time / Epoch (s) | Latency (ms / batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID Score | Kestabilan Training |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Autoencoder** | 64 | 7.72 s | 0.27 ms | **0.0089** | **20.50 dB** | **0.7760** | N/A (Latent Voids) | Stabil / Konvergen |
| **Sparse Autoencoder** | 64 | 6.77 s | 0.38 ms | **0.0088** | **20.58 dB** | **0.7791** | N/A (Latent Voids) | Stabil / Konvergen |
| **Variational Autoencoder (VAE)** | 64 | 8.50 s | 0.33 ms | **0.0081** | **20.91 dB** | **0.7904** | 0.01 | Stabil / Konvergen |
| **Generative Adversarial Network (GAN)** | 64 (Prior Noise) | 11.58 s | 0.96 ms | N/A (Tanpa Encoder) | N/A | N/A | 0.00 | Sangat Rentan Osilasi / Mode Collapse |
| **Deep Convolutional GAN (DCGAN)** | 100 (Prior Noise) | 12.18 s | 0.63 ms | N/A (Tanpa Encoder) | N/A | N/A | 0.00 | Cukup Stabil (Sensitif Arsitektur) |
| **Denoising Diffusion (DDPM)** | 28x28 (Citra) | 16.88 s | 1276.28 ms | N/A (Denoising Model) | N/A | N/A | 0.01 | Sangat Stabil |

---

## 4. Pembuktian Batasan Teoretis & Empiris MLP Classifier (Bagian 2)

Eksperimen pembuktian batasan model klasifikasi terawasi (supervised MLP Classifier) membuktikan tiga batasan matematis utama:

1. **Akurasi Model Klasifikasi (Tahap 1):**  
   Arsitektur $784 \to 256 \to 64 \to 10$ mencapai akurasi test set **88.77%** ($\ge 88.0\%$).
2. **Ketidakbalikan Matriks Bobot ($W^T \ne W^{-1}$) (Tahap 2 & 3):**  
   - Proyeksi balik langsung transpos bobot dari vektor kelas $y_c$ ($\hat{x} = \text{Sigmoid}(W_1^T \text{ReLU}(W_2^T \text{ReLU}(W_3^T y_c)))$) hanya menghasilkan superposisi pola bercak kabur.
   - Proyeksi transpos bobot dari fitur laten citra uji riil $z_{MLP}$ menghasilkan rekonstruksi rusak parah dengan **MSE 0.2781**, **PSNR 5.56 dB**, dan **SSIM 0.0895**. Ini membuktikan secara aljabar linier bahwa forward transformation matriks persegi panjang non-ortogonal ($W^T W \ne \mathbf{I}$) memotong informasi spasial melalui ReLU secara ireversibel.
3. **Information Bottleneck pada Fitur Laten Diskriminatif (Tahap 4):**  
   Ketika MLP dibekukan (`requires_grad=False`) dan dilatihkan Decoder terpisah (64 $\to$ 256 $\to$ 784), hasil rekonstruksi mencapai **MSE 0.0182, PSNR 17.41 dB, SSIM 0.6417**. Performa ini jauh lebih rendah dibandingkan Vanilla Autoencoder berdimensi laten sama ($z \in \mathbb{R}^{64}$) yang mencapai **MSE 0.0089, PSNR 20.50 dB, SSIM 0.7760**. Hal ini membuktikan bahwa objektif Cross-Entropy $P(y|x)$ membuang detail spasial intra-kelas dan hanya mempertahankan fitur pembeda kelas.
4. **Tiadanya Regularisasi Prior Kontinu (Tahap 5):**  
   Penarikan sampel acak $z_{rand} \sim \mathcal{N}(\mu_z, \text{diag}(\sigma_z^2))$ pada ruang laten MLP Classifier diteruskan ke Decoder menghasilkan noise dan artefak acak tanpa struktur busana yang koheren, membuktikan ruang laten MLP tidak membentuk generative manifold yang kontinu.

---

## 5. Ringkasan Lima Figur Visualisasi Komparatif (Bagian V)

Berkas visualisasi resolusi tinggi (300 DPI) telah disimpan di folder `outputs/figures/`:

1. **Figur 1: `figur1_rekonstruksi_side_by_side.png`**  
   Menampilkan perbandingan side-by-side 10 citra uji: Baris 1 (Ground Truth), Baris 2 (Vanilla AE), Baris 3 (Sparse AE), Baris 4 (VAE), dan Baris 5 (Frozen Latent Decoder MLP). Memperlihatkan perbedaan ketajaman kontur, efek keburaman loss rekonstruksi, serta hilangnya tekstur pada MLP latent decoder.
2. **Figur 2: `figur2_random_generative_sampling.png`**  
   Menampilkan kisi 4x4 (16 sampel) sintesis acak: Panel A (VAE), Panel B (Vanilla GAN), Panel C (DCGAN), dan Panel D (DDPM). Menggambarkan mode coverage VAE yang beragam namun halus, ketajaman kontur pada DCGAN, serta sintesis bertahap DDPM.
3. **Figur 3: `figur3_bukti_batasan_mlp_classifier.png`**  
   Menampilkan empat panel bukti empiris: (1) Ground Truth citra uji, (2) Proyeksi transpos bobot $W^T$, (3) Rekonstruksi Frozen Latent Decoder vs Vanilla AE, dan (4) Random prior sampling pada ruang laten MLP.
4. **Figur 4: `figur4_linear_latent_interpolation.png`**  
   Menampilkan transisi linier $z(\alpha) = (1-\alpha)z_A + \alpha z_B$ untuk $\alpha \in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]$ pada Vanilla AE, Sparse AE, VAE, Vanilla GAN, dan DCGAN. Menunjukkan transisi halus dan kontinuitas manifold pada VAE dibandingkan artefak visual (ghosting/latent voids) pada AE deterministik.
5. **Figur 5: `figur5_training_loss_curves_dan_ddpm_trajectory.png`**  
   Menampilkan kurva loss pelatihan seluruh model serta visualisasi sekuens reverse diffusion DDPM langkah demi langkah dari $x_T$ (pure Gaussian noise) menuju $x_0$ (clean generated image).

---

## 6. Referensi Jawaban Pertanyaan Penelitian Ilmiah (Q1 - Q6)

- **Q1 (Fidelitas Rekonstruksi vs Kualitas Generatif):**  
  VAE dan Autoencoder unggul dalam preservasi struktur global citra uji ($SSIM \approx 0.78 - 0.79$). Regularisasi sparsity L1 ($\lambda \|z\|_1$) pada Sparse AE mendorong selektivitas aktivasi neuron tersembunyi, sehingga fitur yang dipertahankan merupakan basis fitur representatif utama dengan fidelitas rekonstruksi yang tetap tinggi ($PSNR = 20.58\text{ dB}$). Untuk generasi sampel baru dari derau murni, DCGAN dan DDPM menghasilkan sampel dengan ketajaman tepi dan variasi mode yang superior dibandingkan Vanilla GAN yang rentan mode collapse.
- **Q2 (Blurriness vs Sharp Edges):**  
  Pixel-wise loss (MSE) meminimalkan $\mathbb{E}[\|x - \hat{x}\|^2]$ yang secara matematis menghasilkan ekspektasi kondisional $\mathbb{E}[x|z]$. Pada distribusi data multimodal, rata-rata dari banyak konfigurasi piksel yang mungkin menghasilkan citra yang buram (blurry). Sebaliknya, adversarial minimax loss pada GAN meminimalkan divergensi Jensen-Shannon $D_{JS}(P_{data} \parallel P_g)$, menuntut generator menghasilkan sampel yang berada tepat pada manifold data riil, sehingga menghasilkan tepi dan kontur yang tajam.
- **Q3 (Stabilitas Pelatihan & Generative Trilemma):**  
  DCGAN memitigasi vanishing gradient pada Vanilla GAN dengan mengganti lapisan fully-connected dengan strided convolutions dan batch normalization. DDPM menggantikan permainan minimax dua pemain yang rentan osilasi dengan supervised noise prediction objective $\mathcal{L}_{simple} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$ yang sangat stabil dan konvergen secara monotonik. Trade-off **Generative Trilemma** terbukti secara empiris: Generator satu langkah (DCGAN, VAE) memiliki kecepatan inferensi tinggi ($<1\text{ ms}$ per batch 64), namun DDPM memerlukan multi-step iterative denoising ($T=200$) dengan latensi $\approx 1276\text{ ms}$ per batch untuk mencapai kualitas sampel dan mode coverage yang maksimal.
- **Q4 (Struktur Latent Space & Interpolasi):**  
  Pada VAE, term KL Divergence memaksakan distribusi posterior $q_\phi(z|x)$ mendekati prior isotropik $\mathcal{N}(0, \mathbf{I})$, mencegah terbentuknya rongga kosong (latent voids) dan memastikan keteraturan (smoothness & continuity) manifold laten saat dilakukan interpolasi linier. Sebaliknya, Autoencoder deterministik (Vanilla & Sparse AE) tidak memiliki regularisasi kepadatan probabilitas, sehingga jalur interpolasi melewati ruang kosong yang memicu timbulnya distorsi bentuk dan artefak ghosting.
- **Q5 (Batasan Model Klasifikasi untuk Image Generation):**  
  Pengujian empiris membuktikan bahwa MLP Classifier gagal men-generate citra karena: (a) $P(y|x)$ menerapkan information bottleneck yang mereduksi variasi spasial intra-kelas hanya menjadi pemisah batas keputusan kelas; (b) Matriks bobot $W$ bersifat non-ortogonal ($W^T W \ne \mathbf{I}$) dan aktivasi ReLU bersifat non-injektif (memotong nilai negatif ke nol); (c) Ruang laten $z_{MLP}$ tidak diikat ke prior probabilitas kontinu $P(z)$.
- **Q6 (Rekomendasi Industri):**  
  1. *Deteksi anomali manufaktur:* **Sparse Autoencoder**, karena sensitivitas penalti sparsity membatasi rekonstruksi hanya pada pola normal, sehingga citra anomali menghasilkan reconstruction error yang jauh lebih tinggi dan mudah dideteksi.
  2. *Filter avatar waktu-nyata mobile (<30 ms):* **DCGAN** atau **VAE**, karena inferensi single-pass forward memiliki latensi sangat rendah ($<1\text{ ms}$ per batch).
  3. *Aset visual resolusi tinggi industri kreatif:* **DDPM (Diffusion Models)**, karena stabilitas pelatihan yang tidak rentan mode collapse serta kemampuan sintesis detail fotorealistik yang komprehensif.
