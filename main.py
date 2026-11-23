import argparse
import sys
import os

from src.experiments_mlp_limits import run_mlp_limitation_experiments
from src.train_generative_models import train_all_models
from src.generate_figures import generate_all_figures

def main():
    parser = argparse.ArgumentParser(
        description="Eksperimen Komparatif Empiris Model Generatif & Pembuktian Batasan MLP Classifier (Deep Learning D6421 Sesi 07)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "mlp_limits", "train_generative", "figures"],
        help="Mode eksekusi: 'all' (menjalankan semua), 'mlp_limits' (Bagian 2), 'train_generative' (Bagian 1 & 4), 'figures' (Bagian 5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Perangkat komputasi: 'cuda' atau 'cpu' (default otomatis mendeteksi CUDA)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("PROYEK TAKE-HOME DEEP LEARNING (D6421) SESI 07")
    print("Studi Komparatif Empiris Model Generatif & Pembuktian Batasan MLP Classifier")
    print("=" * 80)

    if args.mode in ["all", "mlp_limits"]:
        print("\n>>> Menjalankan Bagian 2: Pembuktian Batasan MLP Classifier...")
        run_mlp_limitation_experiments(device=args.device)

    if args.mode in ["all", "train_generative"]:
        print("\n>>> Menjalankan Pelatihan & Evaluasi 6 Model Generatif...")
        train_all_models(device=args.device)

    if args.mode in ["all", "figures"]:
        print("\n>>> Menghasilkan 5 Figur Visualisasi Wajib (300 DPI)...")
        generate_all_figures()

    print("\n" + "=" * 80)
    print("SELURUH RANGKAIAN EKSPERIMEN TELAH SELESAI DENGAN SUKSES!")
    print("Checkpoints tersimpan di: ./checkpoints/")
    print("Metrik dan kurva tersimpan di: ./outputs/")
    print("Figur visualisasi tersimpan di: ./outputs/figures/")
    print("=" * 80)

if __name__ == "__main__":
    main()
