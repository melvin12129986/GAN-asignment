# LAPORAN TUGAS TAKE HOME DEEP LEARNING (D6421)
## STUDI KOMPARATIF MODEL GENERATIF: EVALUASI EMPIRIS AUTOENCODER, VARIATIONAL AUTOENCODER, GENERATIVE ADVERSARIAL NETWORK, DAN DENOISING DIFFUSION PROBABILISTIC MODEL PADA DATASET FASHION-MNIST

---

* **Mata Kuliah:** Deep Learning (D6421) — Sesi 07
* **Topik:** Generative Models & Representation Learning
* **Dataset Benchmark:** Fashion-MNIST (60.000 citra $28 \times 28$ grayscale, 10 kelas busana)
* **Tautan Repositori GitHub:** [https://github.com/melvin12129986/GAN-asignment](https://github.com/melvin12129986/GAN-asignment)

---

## DAFTAR ISI
1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Metodologi & Pipeline Pelatihan](#2-metodologi--pipeline-pelatihan)
3. [Pembuktian Batasan Teoretis & Empiris MLP Classifier (Bagian II)](#3-pembuktian-batasan-teoretis--empiris-mlp-classifier-bagian-ii)
4. [Tabel Komparasi Kuantitatif Benchmark (Bagian IV)](#4-tabel-komparasi-kuantitatif-benchmark-bagian-iv)
5. [Analisis Visual & Pembahasan Lima Figur Komparatif 300 DPI (Bagian V)](#5-analisis-visual--pembahasan-lima-figur-komparatif-300-dpi-bagian-v)
6. [Analisis Jawaban Pertanyaan Ilmiah Mendalam (Q1 – Q6)](#6-analisis-jawaban-pertanyaan-ilmiah-mendalam-q1--q6)
7. [Kesimpulan](#7-kesimpulan)

---

## 1. RINGKASAN EKSEKUTIF

Laporan ini memuat evaluasi komprehensif terhadap enam arsitektur model generatif terkemuka—**Vanilla Autoencoder (AE)**, **Sparse Autoencoder (SAE)**, **Variational Autoencoder (VAE)**, **Vanilla GAN (MLP)**, **Deep Convolutional GAN (DCGAN)**, dan **Denoising Diffusion Probabilistic Model (DDPM)**—serta satu arsitektur **Multilayer Perceptron (MLP) Classifier** sebagai model dasar diskriminatif terawasi. 

Eksperimen dilakukan di bawah protokol yang terstandarisasi ketat menggunakan pustaka PyTorch 2.6.0 dengan akselerasi GPU NVIDIA RTX 3060 Laptop (CUDA 12.4). Pembagian dataset Fashion-MNIST mengikuti rasio **80% data latih (56.000)**, **10% data validasi (7.000)**, dan **10% data uji (7.000)** dengan penguncian seed deterministik (`seed=42`). 

Eksperimen membuktikan dua tesis utama:
1. Model klasifikasi diskriminatif menderita fenomena *Information Bottleneck* dan hilangnya invertibilitas matriks aljabar linier, sehingga gagal total saat dipaksa bertindak sebagai generator citra.
2. Terdapat trade-off yang tidak dapat dihindari (*Generative Trilemma*) antara kesempurnaan fidelitas rekonstruksi sampel uji, ketajaman perseptual generasi sampel baru (FID), dan kecepatan inferensi (latensi).

---

## 2. METODOLOGI & PIPELINE PELATIHAN

### 2.1 Konfigurasi Dataset & Preprocessing
Dataset Fashion-MNIST dinormalisasi ke dalam rentang kontras $[0.0, 1.0]$. Untuk menjamin keabsahan pengujian tanpa kebocoran data (*data leakage*):
* **Data Latih:** 56.000 sampel (digunakan untuk memperbarui gradien parameter jaringan).
* **Data Validasi:** 7.000 sampel (digunakan untuk pemantauan konvergensi).
* **Data Uji:** 7.000 sampel (disimpan secara terisolasi hingga tahap evaluasi final metrik MSE, PSNR, SSIM, dan FID).

### 2.2 Arsitektur Model yang Diuji
1. **MLP Classifier:** Jaringan sekuensial $784 \to 256 \to 64 \to 10$. Lapisan tersembunyi menggunakan aktivasi ReLU, diakhiri dengan logits 10-kelas dengan fungsi rugi *Cross-Entropy Loss*.
2. **Vanilla Autoencoder:** Arsitektur simetris dengan Encoder $784 \to 256 \to 64$ dan Decoder $64 \to 256 \to 784$ (aktivasi akhir Sigmoid). Fungsi rugi: *Mean Squared Error (MSE)*.
3. **Sparse Autoencoder:** Memiliki struktur identik dengan Vanilla AE, namun dilengkapi penalti kelangkaan (*L1 Sparsity Penalty*) pada representasi laten: $\mathcal{L}_{SAE} = \mathcal{L}_{MSE} + \lambda \|z\|_1$, dengan $\lambda = 10^{-4}$.
4. **Variational Autoencoder (VAE):** Encoder memproyeksikan citra input menjadi parameter distribusi Gaussian multivariat $(\mu, \log \sigma^2) \in \mathbb{R}^{64}$. Vektor laten diperoleh melalui *Reparameterization Trick* $z = \mu + \sigma \odot \epsilon$, di mana $\epsilon \sim \mathcal{N}(0, \mathbf{I})$. Fungsi objektif adalah ELBO (*Evidence Lower Bound*): $\mathcal{L}_{VAE} = \mathcal{L}_{MSE} + \beta \mathcal{D}_{KL}(q_\phi(z|x) \parallel \mathcal{N}(0, \mathbf{I}))$.
5. **Vanilla GAN (MLP):** Generator $64 \to 256 \to 512 \to 784$ dan Discriminator $784 \to 512 \to 256 \to 1$ dilatih menggunakan permainan minimax dua pemain dengan fungsi rugi *Binary Cross-Entropy (BCEWithLogitsLoss)* dan teknik *one-sided label smoothing* ($0.9$ untuk real).
6. **DCGAN (Deep Convolutional GAN):** Generator memanfaatkan *Fractionally-strided Convolutions (ConvTranspose2d)* dengan normalisasi *BatchNorm2d* dan aktivasi *ReLU*, memetakan prior laten $z \in \mathbb{R}^{100}$ menjadi citra $28 \times 28 \times 1$. Discriminator menggunakan *Strided Convolutions (Conv2d)* dengan *LeakyReLU (slope 0.2)*.
7. **DDPM (Denoising Diffusion Probabilistic Model):** Arsitektur U-Net dengan residual convolutional blocks dan *Sinusoidal Position Embeddings* untuk menyuntikkan informasi langkah waktu $t \in [1, T]$ ($T = 200$). Jadwal derau linier $\beta_1 = 10^{-4}$ hingga $\beta_T = 0.02$. Model memprediksi komponen noise $\epsilon_\theta(x_t, t)$ menggunakan supervised loss $\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$.

---

## 3. PEMBUKTIAN BATASAN TEORETIS & EMPIRIS MLP CLASSIFIER (BAGIAN II)

Modul praktikum menuntut pembuktian sistematis mengapa model supervised classification tidak dapat ditransformasikan menjadi generative model. Pembuktian dilakukan dalam 5 tahap empiris berurutan:

```
[Tahap 1: Train MLP] ──> Akurasi 88.77% (Baseline Diskriminatif Valid)
         │
         ├──> [Tahap 2: W^T y_c] ──────────> Citra blob kabur tanpa struktur
         │
         ├──> [Tahap 3: W^T z_MLP] ────────> MSE: 0.2781, PSNR: 5.56 dB, SSIM: 0.0895 (Rusak Total)
         │
         ├──> [Tahap 4: Frozen Decoder] ───> MSE: 0.0182 (Kalah 2x lipat vs AE 0.0089)
         │
         └──> [Tahap 5: Prior Sampling] ───> Noise acak / tidak koheren (Latent Voids)
```

### Tahap 1: Verifikasi Akurasi Klasifikasi
* **Metrik:** Akurasi data latih: $90.84\%$, Akurasi validasi: $88.54\%$, Akurasi data uji final: **$88.77\%$** (melampaui ambang batas syarat kelulusan $\ge 88.0\%$).
* **Kesimpulan:** Model klasifikasi telah mencapai tingkat pemisahan batas keputusan (*decision boundary*) yang optimal pada ruang fitur Fashion-MNIST.

### Tahap 2: Proyeksi Transpos Bobot dari Vektor Kelas (Top-Down Weight Transposition)
* **Formulasi:** Dari vektor one-hot kelas $y_c \in \mathbb{R}^{10}$, dilakukan proyeksi balik: $\hat{x} = \sigma(W_1^T \text{ReLU}(W_2^T \text{ReLU}(W_3^T y_c)))$.
* **Hasil Empiris:** Citra yang dihasilkan untuk kesepuluh kelas busana hanya berupa superposisi intensitas piksel rata-rata yang menyerupai gumpalan kabur (*blurry blob*) tanpa tepian tegas maupun tekstur kain.

### Tahap 3: Proyeksi Transpos Bobot dari Latent Citra Riil (Latent Transpose Projection)
* **Formulasi:** Citra uji riil dilewatkan ke MLP hingga menghasilkan representasi laten $z_{MLP} \in \mathbb{R}^{64}$, lalu direkonstruksi: $\hat{x} = \sigma(W_1^T \text{ReLU}(W_2^T z_{MLP}))$.
* **Hasil Empiris:** Diperoleh **MSE = 0.2781**, **PSNR = 5.56 dB**, dan **SSIM = 0.0895**.
* **Analisis Teoretis Aljabar Linier:** Matriks pembobotan $W_1 \in \mathbb{R}^{256 \times 784}$ dan $W_2 \in \mathbb{R}^{64 \times 256}$ merupakan matriks non-bujursangkar dengan dimensi input yang jauh lebih besar daripada dimensi output. Operasi transpos matriks $W^T$ bukanlah invers matriks ($W^T \ne W^{-1}$). Karena $W^T W \ne \mathbf{I}$, transformasi maju bersifat non-injektif dan mereduksi rank matriks (*rank-deficient*). Selain itu, fungsi aktivasi non-linier $\text{ReLU}(a) = \max(0, a)$ memotong seluruh komponen gradien negatif ke nol, mengakibatkan pemusnahan informasi spasial secara permanen (*loss of information*).

### Tahap 4: Information Bottleneck pada Fitur Laten (Frozen Latent Decoding)
* **Formulasi:** Seluruh bobot MLP Classifier dibekukan secara permanen (`requires_grad=False`). Decoder terpisah berkapasitas penuh ($64 \to 256 \to 784$) dilatih dengan loss MSE untuk merekonstruksi citra asli dari $z_{MLP}$.
* **Hasil Kuantitatif Komparatif:**
  * Frozen Latent Decoder: **MSE = 0.0182**, **PSNR = 17.41 dB**, **SSIM = 0.6417**.
  * Vanilla Autoencoder ($z \in \mathbb{R}^{64}$): **MSE = 0.0089**, **PSNR = 20.50 dB**, **SSIM = 0.7760**.
* **Analisis Prinsip Information Bottleneck (Tishby et al.):** Objective klasifikasi Cross-Entropy memaksimalkan informasi timbal balik $I(Z; Y)$ sembari meminimalkan $I(X; Z)$. Akibatnya, representasi laten $z_{MLP}$ secara agresif membuang seluruh variasi intra-kelas (tekstur rajutan, lipatan kain, kancing, gradasi bayangan) dan hanya melestarikan atribut ortogonal pembeda antar-kelas. Meskipun Decoder dilatih hingga konvergen, informasi geometris mikro tidak lagi tersedia di dalam $z_{MLP}$.

### Tahap 5: Ketiadaan Regularisasi Ruang Laten (Prior Sampling)
* **Formulasi:** Menghitung vektor rata-rata empiris $\mu_z = \mathbb{E}[z_{MLP}]$ dan kovariansi $\sigma_z^2 = \text{Var}[z_{MLP}]$, kemudian menarik sampel acak $z_{rand} \sim \mathcal{N}(\mu_z, \text{diag}(\sigma_z^2))$ lalu menyalurkannya ke Frozen Latent Decoder.
* **Hasil Empiris:** Citra yang disintesis berupa noise statis acak, pola garis retak, dan artefak kabur tanpa wujud busana yang koheren.
* **Analisis Topologi Manifold:** Ruang laten classifier tidak dikenakan regularisasi kepadatan probabilitas kontinu. Akibatnya, manifold laten terfragmentasi menjadi kluster-kluster diskrit yang dipisahkan oleh jurang kekosongan (*latent voids*). Menarik sampel dari ruang kosong tersebut memicu decoder mengekstrapolasi representasi di luar domain distribusinya.

---

## 4. TABEL KOMPARASI KUANTITATIF BENCHMARK (BAGIAN IV)

Tabel berikut menyajikan ringkasan metrik kuantitatif hasil evaluasi seluruh model pada Fashion-MNIST Test Set (7.000 citra uji). Metrik FID dihitung menggunakan **InceptionV3 Pretrained ImageNet (pool3 layer, 2048-dimensi)** dengan membandingkan 2.048 citra uji riil terhadap 2.048 citra sintetis masing-masing model generatif:

| Arsitektur Model | Dim Laten | Train Time / Epoch (s) | Latency (ms / batch 64) | Test MSE ↓ | Test PSNR (dB) ↑ | Test SSIM ↑ | FID Score ↓ | Kestabilan Training |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Autoencoder** | 64 | 7.72 s | 0.27 ms | **0.0089** | **20.50 dB** | **0.7760** | *N/A (Latent Voids)* | Stabil / Konvergen Monotonik |
| **Sparse Autoencoder** | 64 | 6.77 s | 0.38 ms | **0.0088** | **20.58 dB** | **0.7791** | *N/A (Latent Voids)* | Stabil / Konvergen Monotonik |
| **Variational Autoencoder (VAE)** | 64 | 8.50 s | 0.33 ms | **0.0081** | **20.91 dB** | **0.7904** | 167.93 | Sangat Stabil (Optimasi ELBO) |
| **Vanilla GAN (MLP)** | 64 | 11.58 s | 0.96 ms | *N/A* | *N/A* | *N/A* | 145.61 | Rentan Osilasi & Mode Collapse |
| **Deep Convolutional GAN (DCGAN)**| 100 | 12.18 s | 0.63 ms | *N/A* | *N/A* | *N/A* | **55.53** | Cukup Stabil (Sensitif Arsitektur) |
| **Denoising Diffusion (DDPM)** | 28×28 | 16.88 s | 1276.28 ms | *N/A* | *N/A* | *N/A* | **65.03** | Sangat Stabil (MSE Noise Pred) |

---

## 5. ANALISIS VISUAL & PEMBAHASAN LIMA FIGUR KOMPARATIF 300 DPI (BAGIAN V)

Seluruh visualisasi telah diekspor dalam resolusi standar publikasi ilmiah (300 DPI) di dalam direktori `outputs/figures/`:

```
outputs/figures/
├── figur1_rekonstruksi_side_by_side.png
├── figur2_random_generative_sampling.png
├── figur3_bukti_batasan_mlp_classifier.png
├── figur4_linear_latent_interpolation.png
└── figur5_training_loss_curves_dan_ddpm_trajectory.png
```

### 5.1 Figur 1: Grid Rekonstruksi Side-by-Side
* **Deskripsi Visual:** Menyajikan perbandingan 10 kolom citra uji Fashion-MNIST dengan 5 baris:
  * Baris 1: Citra Asli (Ground Truth).
  * Baris 2: Rekonstruksi Vanilla Autoencoder.
  * Baris 3: Rekonstruksi Sparse Autoencoder.
  * Baris 4: Rekonstruksi Variational Autoencoder (VAE).
  * Baris 5: Rekonstruksi Frozen Latent Decoder (MLP Classifier).
* **Temuan Analitis:** 
  Vanilla AE dan Sparse AE merekonstruksi siluet pakaian dengan kontur tajam dan rasio MSE sangat rendah ($\approx 0.0088$). VAE menghasilkan rekonstruksi yang sedikit lebih lembut (*smooth*) akibat penalti keteraturan KL Divergence. Sebaliknya, Baris 5 (MLP Latent Decoder) memperlihatkan hilangnya detail tekstur mikro (misalnya detail tali sepatu sneaker dan motif kerah baju) serta degradasi ketajaman tepi, yang mengonfirmasi bahwa ruang laten diskriminatif membuang variasi intra-kelas.

### 5.2 Figur 2: Grid Random Generative Sampling (4-Panel)
* **Deskripsi Visual:** Menampilkan sintesis 16 sampel acak ($4 \times 4$ grid) untuk 4 model generatif murni: Panel A (VAE), Panel B (Vanilla GAN), Panel C (DCGAN), dan Panel D (DDPM).
* **Temuan Analitis:**
  * *VAE (Panel A):* Memiliki keragaman kelas busana (*mode coverage*) yang sangat tinggi dan merata, namun seluruh citra memperlihatkan keburaman struktural (*structural blurriness*).
  * *Vanilla GAN (Panel B):* Menghasilkan citra dengan artefak distorsi lokal dan indikasi awal parsial *mode collapse*, di mana beberapa bentuk sepatu dan tas mengalami peleburan bentuk (*shape morphing* yang anomali).
  * *DCGAN (Panel C):* Menunjukkan garis batas yang sangat tegas dan kontras tinggi pada pakaian, tas tangan, dan sepatu boot, membuktikan efektivitas inductive bias dari konvolusi spasial.
  * *DDPM (Panel D):* Menghasilkan sampel dengan variasi kelas yang komprehensif, kontur realistis, dan latar belakang hitam bersih bebas artefak catur (*checkerboard artifacts*).

### 5.3 Figur 3: Bukti Empiris Batasan Bobot dan Latent MLP Classifier
* **Deskripsi Visual:** Terdiri atas 4 sub-panel: Panel (a) Ground Truth, Panel (b) Proyeksi Transpos Bobot Langsung $W^T y_c$, Panel (c) Rekonstruksi Frozen Latent Decoder vs Vanilla AE, dan Panel (d) Prior Sampling pada Ruang Laten MLP.
* **Temuan Analitis:** Figur ini menjadi bukti empiris visual yang mengonfirmasi kegagalan matematis supervised classifier untuk bertindak sebagai generator. Panel (b) memperlihatkan citra seragam tanpa kontur yang membuktikan $W^T \ne W^{-1}$. Panel (d) memperlihatkan kumpulan derau statis akibat kekosongan manifold laten (*latent voids*).

### 5.4 Figur 4: Linear Latent Space Interpolation
* **Deskripsi Visual:** Melakukan interpolasi linier $z(\alpha) = (1-\alpha)z_A + \alpha z_B$ dengan 6 titik transisi $\alpha \in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]$ antara dua busana yang berbeda secara topologi (misalnya dari Gaun/Dress ke Sepatu Boot/Ankle Boot).
* **Temuan Analitis:**
  * *VAE:* Memperlihatkan transisi bentuk yang sangat kontinu, mulus, dan semantik. Di titik tengah ($\alpha=0.4 - 0.6$), objek berevolusi secara organik tanpa menghasilkan citra yang rusak.
  * *Vanilla & Sparse AE:* Mengalami fenomena *ghosting* atau superposisi transparan di titik tengah $\alpha = 0.5$. Citra gaun memudar perlahan sementara citra sepatu boot muncul secara bertumpuk. Hal ini membuktikan lintasan interpolasi melewati rongga kosong yang tidak pernah dipetakan oleh data riil.
  * *DCGAN:* Mampu melakukan interpolasi mulus pada orientasi dan ketebalan objek busana, membuktikan bahwa generator konvolusional mempelajari representasi manifold kontinu dari distribusi prior Gaussian.

### 5.5 Figur 5: Training Loss Curves & DDPM Denoising Trajectory
* **Deskripsi Visual:** Sisi kiri menampilkan dinamika kurva konvergensi fungsi rugi pelatihan untuk seluruh model. Sisi kanan menampilkan visualisasi proses *reverse diffusion* DDPM langkah demi langkah dari $t=200$ (pure Gaussian noise $\mathcal{N}(0, \mathbf{I})$) hingga $t=0$ (citra busana sintesis yang jernih).
* **Temuan Analitis:**
  * Kurva loss AE, VAE, dan DDPM konvergen secara monotonik dan stabil.
  * Kurva loss GAN (Generator vs Discriminator) memperlihatkan osilasi kompetitif khas permainan minimax minimax zero-sum, di mana penurunan loss Discriminator sering kali memicu lonjakan loss Generator.
  * Lintasan DDPM membuktikan mekanisme pembentukan struktur secara bertahap: pada interval $t \in [200, 150]$, model menetapkan siluet kasar objek; pada $t \in [150, 50]$, kontur bentuk dikristalisasi; dan pada $t \in [50, 0]$, detail frekuensi tinggi dan penghapusan noise sisa diselesaikan.

---

## 6. ANALISIS JAWABAN PERTANYAAN ILMIAH MENDALAM (Q1 – Q6)

### Pertanyaan 1 (Q1): Fidelitas Rekonstruksi vs Kualitas Generatif
* **Pertanyaan:** Mengapa model dengan performa rekonstruksi terbaik (seperti Autoencoder) belum tentu menjadi model terbaik untuk membangkitkan sampel baru dari derau acak?
* **Analisis Jawaban:**  
  Tujuan matematis Autoencoder deterministik adalah meminimalkan $\mathbb{E}_{x \sim P_{data}}[\|x - g_\theta(f_\phi(x))\|^2]$. Ini adalah pemetaan diskrit titik-ke-titik (*point-to-point mapping*). Encoder hanya menempatkan data latih pada koordinat-koordinat spesifik di $\mathbb{R}^{64}$ tanpa adanya batasan geometris pada ruang di antara koordinat-koordinat tersebut. Akibatnya, ruang laten dipenuhi oleh rongga hampa (*latent voids*) di mana densitas probabilitas $P(z) \approx 0$. Saat penarikan sampel acak dilakukan dari prior standar $z \sim \mathcal{N}(0, \mathbf{I})$, probabilitas mendarat di koordinat hampa tersebut sangat tinggi, sehingga Decoder menghasilkan output yang tidak masuk akal (*nonsense/unrealistic samples*). Sebaliknya, model generatif sejati seperti DCGAN dan DDPM tidak pernah dioptimalkan untuk merekonstruksi citra titik-ke-titik, melainkan dilatih untuk mencocokkan distribusi densitas probabilitas secara keseluruhan ($P_g(x) \approx P_{data}(x)$), sehingga seluruh wilayah pada ruang prior dapat dipetakan menjadi sampel sintetis yang valid.

### Pertanyaan 2 (Q2): Blurriness vs Sharp Edges (MSE vs Adversarial Loss)
* **Pertanyaan:** Jelaskan fenomena keburaman citra (*blurriness*) pada VAE dan Autoencoder dibandingkan ketajaman tepian (*sharp edges*) pada GAN ditinjau dari formulasi fungsi rugi masing-masing.
* **Analisis Jawaban:**  
  Fungsi rugi piksel berbasis *Mean Squared Error (MSE)* diturunkan dari asumsi distribusi kemungkinan maksimum (*Maximum Likelihood Estimation*) dengan model noise Gaussian isotropik: $-\log P(x|z) \propto \frac{1}{2\sigma^2}\|x - \hat{x}\|^2$. Ketika terdapat ambiguitas atau ketidakpastian multimodal pada data (misalnya posisi saku baju atau lipatan celana yang bervariasi di antara sampel latih), solusi optimal yang meminimalkan MSE secara analitis adalah nilai rata-rata kondisional (*conditional mean*):
  $$\hat{x}^* = \arg\min_{\hat{x}} \mathbb{E}[\|x - \hat{x}\|^2] = \mathbb{E}_{P(x|z)}[x]$$
  Rata-rata dari berbagai kemungkinan konfigurasi piksel menghasilkan efek penghalusan (*smoothing/averaging effect*) yang tampak sebagai citra kabur.  
  Sebaliknya, GAN menggunakan fungsi rugi adversarial yang meminimalkan Divergensi Jensen-Shannon:
  $$\mathcal{L}_{GAN} = 2 D_{JS}(P_{data} \parallel P_g) - 2\log 2$$
  Discriminator bertindak sebagai penilai perseptual non-linier terparameterisasi yang secara aktif menghukum citra rata-rata yang buram karena citra buram tersebut tidak berada pada manifold data riil ($D(x_{blur}) \to 0$). Agar mampu mengecoh Discriminator, Generator dipaksa memilih satu mode spesifik yang tajam dan menghasilkan gradien tepi frekuensi tinggi (*high-frequency sharp edges*).

### Pertanyaan 3 (Q3): Stabilitas Pelatihan & Generative Trilemma
* **Pertanyaan:** Bandingkan kestabilan pelatihan DCGAN dan DDPM, serta kaitkan hasil empiris Anda dengan konsep *Generative Trilemma*.
* **Analisis Jawaban:**  
  DCGAN melatih dua jaringan saraf secara simultan dalam formulasi optimasi minimax $\min_G \max_D V(D, G)$. Keseimbangan Nash (*Nash Equilibrium*) pada ruang parameter non-konveks berdimensi tinggi sangat sulit dicapai, sehingga pelatihan rentan mengalami osilasi gradien, ketidakstabilan numerik (*exploding/vanishing gradients*), dan *mode collapse*. Sebaliknya, DDPM mentransformasikan masalah generatif menjadi estimasi noise terawasi (*supervised denoising objective*) $\mathcal{L}_{simple} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$. Dinamika pelatihan DDPM sepenuhnya monotonik, konvergen mulus tanpa melibatkan interaksi kompetitif dua pemain.  
  Hasil eksperimen ini secara gamblang membuktikan **Generative Trilemma**, yaitu kompromi mendasar antara tiga pilar model generatif:
  1. *Kecepatan Inferensi Tinggi (Fast Sampling):* DCGAN ($0.63\text{ ms}$) dan VAE ($0.33\text{ ms}$) memerlukan hanya satu langkah komputasi maju (*single forward pass*).
  2. *Kualitas Visual Tinggi (High Sample Quality):* DCGAN (FID $55.53$) dan DDPM (FID $65.03$) menghasilkan citra dengan fidelity perseptual superior.
  3. *Cakupan Mode Penuh (Mode Coverage / Diversity):* DDPM dan VAE mencakup seluruh ragam distribusi data tanpa risiko mode collapse.
  
  DDPM mengorbankan kecepatan inferensi secara signifikan demi meraih kualitas visual dan cakupan mode yang optimal: latensi DDPM pada batch 64 mencapai **$1276.28\text{ ms}$** (sekitar $2000\times$ lebih lambat daripada DCGAN) karena harus mengeksekusi $T=200$ langkah iteratif pembalikan derau.

### Pertanyaan 4 (Q4): Struktur Latent Space & Pengaruh Regularisasi KL
* **Pertanyaan:** Mengapa interpolasi linier pada VAE menghasilkan transisi bentuk yang halus dan alami, sedangkan Autoencoder deterministik menghasilkan artefak visual?
* **Analisis Jawaban:**  
  Fungsi objektif VAE menyertakan suku penalti divergensi Kullback-Leibler:
  $$\mathcal{D}_{KL}(q_\phi(z|x) \parallel \mathcal{N}(0, \mathbf{I})) = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$
  Term KL Divergence memberikan dua sifat geometris esensial pada ruang laten:
  1. *Kontinuitas (Continuity):* Titik-titik laten yang berdekatan menghasilkan output rekonstruksi yang serupa.
  2. *Kelengkapan (Completeness):* Seluruh area di sekitar pusat distribusi memetakan representasi data yang bermakna (*no empty holes*).  
  Ketika vektor laten diinterpolasi secara linier $z(\alpha) = (1-\alpha)z_A + \alpha z_B$, lintasan interpolasi dijamin selalu melintasi wilayah dengan densitas probabilitas tinggi di bawah prior $\mathcal{N}(0, \mathbf{I})$. Pada Autoencoder deterministik tanpa regularisasi KL, ruang laten tidak memiliki batasan kontinuitas. Lintasan interpolasi melintasi *latent voids* yang tidak berkorespondensi dengan data riil, memicu Decoder mengekstrapolasi bentuk hibrida yang terdistorsi (*ghosting/fading artifacts*).

### Pertanyaan 5 (Q5): Mengapa Model Klasifikasi Gagal Menjadi Model Generatif?
* **Pertanyaan:** Rangkum faktor penyebab mendasar ketidakmampuan MLP Classifier dalam membangkitkan citra berdasarkan hasil pembuktian Tahap 1 hingga Tahap 5.
* **Analisis Jawaban:**  
  Kegagalan fundamental model klasifikasi terawasi berakar pada tiga alasan:
  1. *Information Bottleneck Teoretis:* Klasifikasi mengoptimalkan batas keputusan posterior $P(Y|X)$. Model secara proaktif memusnahkan informasi struktural piksel intrakelas yang tidak esensial untuk membedakan label, menyisakan hanya fitur diskriminatif ortogonal.
  2. *Ireversibilitas Aljabar Linier:* Matriks bobot $W$ bersifat non-bujursangkar dan memproyeksikan data dari ruang dimensi tinggi ($784$) ke dimensi rendah ($10$). Matriks transpos $W^T$ bukanlah invers matematis ($W^T W \ne \mathbf{I}$), dan fungsi aktivasi ReLU memotong setengah domain gradien secara ireversibel.
  3. *Ketiadaan Model Kepadatan Probabilitas (No Prior $P(Z)$):* Model diskriminatif tidak memodelkan distribusi data bersama $P(X)$, sehingga tidak memiliki kerangka probabilitas matematis untuk menarik sampel baru dari ruang fitur laten.

### Pertanyaan 6 (Q6): Rekomendasi Penerapan Industri Nyata
* **Pertanyaan:** Model manakah yang Anda rekomendasikan untuk masing-masing dari tiga skenario industri berikut, dan apa landasan teoretisnya?
  * Skenario A: Deteksi Anomali Defek pada Lini Manufaktur Otomatis.
  * Skenario B: Filter Avatar Augmented Reality Real-Time pada Aplikasi Mobile Gaming ($<30\text{ ms}$).
  * Skenario C: Pembangkitan Aset Tekstil & Busana Resolusi Tinggi untuk Industri Mode Kreatif.
* **Analisis Jawaban:**
  * **Skenario A (Deteksi Anomali Manufaktur) $\to$ SPARSE AUTOENCODER:**  
    *Landasan:* Sparse AE dilatih hanya menggunakan sampel produk normal dengan penalti kelangkaan aktivasi $\lambda \|z\|_1$. Basis fitur laten dipaksa mengompilasi representasi esensial produk tanpa cacat secara sangat efisien. Ketika citra produk cacat/anomali dimasukkan, jaringan tidak memiliki basis fitur untuk merekonstruksi defek tersebut. Akibatnya, peta kesalahan rekonstruksi piksel $\|x - \hat{x}\|^2$ akan melonjak drastis pada koordinat cacat, memungkinkan lokalisasi defek dengan tingkat presisi tinggi tanpa memerlukan anotasi data anomali yang mahal.
  * **Skenario B (Filter Avatar Real-Time Mobile) $\to$ DEEP CONVOLUTIONAL GAN (DCGAN) / VAE:**  
    *Landasan:* Syarat operasional mobile gaming menuntut latensi inferensi di bawah ambang batas persepsi mata manusia ($<30\text{ ms}$). DCGAN dan VAE memiliki struktur komputasi *single forward pass* langsung dari prior laten $z$ menuju citra output, dengan latensi empiris batch sebesar **$0.63\text{ ms}$** (jauh di bawah batas $30\text{ ms}$). DCGAN menjadi pilihan utama jika ketajaman visual diprioritaskan, sedangkan VAE dipilih jika dibutuhkan kontrol interpolasi atribut wajah yang mulus.
  * **Skenario C (Aset Mode Industri Kreatif) $\to$ DENOISING DIFFUSION PROBABILISTIC MODEL (DDPM):**  
    *Landasan:* Pada industri desain kreatif, variasi mode yang luas (*mode diversity*) dan ketiadaan cacat tekstur sintesis merupakan prioritas mutlak di atas kecepatan inferensi. DDPM tidak menderita *mode collapse* (FID stabil sebesar $65.03$), menjamin seluruh katalog gaya busana dapat dieksplorasi secara lengkap dengan detail tekstur tenun, kontur kerah, dan lipatan kain yang fotorealistik.

---

## 7. KESIMPULAN

Berdasarkan keseluruhan rangkaian eksperimen dan analisis empiris yang telah dilakukan:
1. Pembuktian batasan MLP Classifier memperlihatkan secara konklusif bahwa pemodelan generatif membutuhkan perancangan ruang representasi probabilitas yang kontinu dan pemodelan eksplisit distribusi data $P(x)$, bukan sekadar pemisahan batas keputusan kelas diskriminatif $P(y|x)$.
2. Pada spektrum rekonstruksi, Autoencoder dan VAE memberikan tingkat fidelitas tertinggi dengan retensi struktur geometris yang stabil ($SSIM > 0.77$).
3. Pada spektrum sintesis generasi baru, DCGAN memberikan efisiensi kecepatan inferensi terbaik ($0.63\text{ ms}$) dengan ketajaman tepi yang tinggi (FID $55.53$), sementara DDPM menjamin stabilitas pelatihan yang sempurna dan cakupan mode yang paling komprehensif.

---
*Laporan ini disusun secara lengkap dan sistematis untuk memenuhi seluruh kriteria evaluasi tugas take-home Deep Learning (D6421).*
