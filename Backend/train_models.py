"""Skrip Pelatihan & Evaluasi Model Skema Orange (CLI).

Mengimplementasikan alur lengkap skema Orange Data Mining (`SMIDS.ows`):
  [Import Images] -> [Image Embedding: SqueezeNet Orange/ndf, 1000-dim] ->
  [5 Learners: MLP, SVM, RF, kNN, NB] ->
  [Test & Score (Evaluasi)] -> [Confusion Matrix] -> [Simpan Model ke app/models/]

Ekstraksi fitur memakai jalur yang SAMA dengan backend (`app.services` +
`app.utils`), sehingga embedding saat pelatihan identik dengan saat inferensi.

Penggunaan:
    # 1. Inisialisasi model baseline (tanpa dataset, acak) agar server bisa start:
    python train_models.py --init-baseline

    # 2. Latih dari folder citra (subfolder: Normal_Sperm, Abnormal_Sperm, Non-Sperm):
    python train_models.py --data-dir path/to/dataset

    # 3. Latih dari berkas fitur CSV/NPY yang sudah diekstrak:
    python train_models.py --features-file features.npy --labels-file labels.npy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Tambahkan direktori root backend ke path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.config import DATASET_FOLDER_TO_LABEL, get_settings  # noqa: E402
from app.services.models_factory import (  # noqa: E402
    CLASS_NAMES,
    MODEL_SPECS,
    train_and_save_all,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("TrainModels")

# Folder kelas dataset Orange (SMIDS) dan urutan label kanonik.
DATASET_FOLDERS = ["Normal_Sperm", "Abnormal_Sperm", "Non-Sperm"]
EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png")


def evaluate_models(X_train, X_test, y_train, y_test):
    """Evaluasi ke-5 model sesuai modul 'Test and Score' dan 'Confusion Matrix' di Orange."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    print("\n" + "=" * 70)
    print("HASIL EVALUASI MODEL (ORANGE: TEST AND SCORE & CONFUSION MATRIX)")
    print("=" * 70)

    for m_id, spec in MODEL_SPECS.items():
        print(f"\n--- Model: {spec['name']} ({spec['paper_table']}) ---")
        clf = spec["builder"]()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi: {acc * 100:.2f}%\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
        print("Confusion Matrix:")
        print(f"Kelas target: {CLASS_NAMES}")
        for i, row in enumerate(cm):
            print(f"  {CLASS_NAMES[i]:<10}: {row}")


def load_dataset_from_images(
    data_dir: Path,
    output_features_dir: Path | None = None,
    max_per_class: int | None = None,
):
    """Ekstraksi fitur SqueezeNet Orange (ndf) dari folder gambar.

    Jalur ekstraksi identik dengan backend: `buka_dan_preprocess` (resize 227,
    RGB->BGR, kurangi mean Caffe) lalu `FeatureExtractor` (ndf flatten10).

    `max_per_class` membatasi jumlah citra per kelas (berguna untuk uji cepat).
    """
    from app.services.feature_extractor import FeatureExtractor
    from app.utils.image_utils import buka_dan_preprocess

    settings = get_settings()
    extractor = FeatureExtractor(settings)
    if not extractor.siap:
        raise RuntimeError(f"Feature extractor gagal dimuat: {extractor.pesan}")

    fitur_list = []
    labels_list = []

    for folder in DATASET_FOLDERS:
        kelas_dir = data_dir / folder
        if not kelas_dir.exists():
            alt_folder = folder.lower().replace("-", "_")
            if (data_dir / alt_folder).exists():
                kelas_dir = data_dir / alt_folder
                folder = alt_folder
            else:
                logger.warning("Folder kelas '%s' (atau '%s') tidak ditemukan di %s", folder, alt_folder, data_dir)
                continue
        label = DATASET_FOLDER_TO_LABEL.get(folder, folder)

        files = [
            p for p in sorted(kelas_dir.iterdir())
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]
        if max_per_class is not None:
            files = files[:max_per_class]
        logger.info("Mengekstrak %d citra dari kelas '%s' -> '%s'...", len(files), folder, label)

        # Proses per batch agar satu forward pass menangani banyak citra.
        BATCH = 64
        for i in range(0, len(files), BATCH):
            chunk = files[i:i + BATCH]
            tensors = []
            for fpath in chunk:
                try:
                    tensors.append(
                        buka_dan_preprocess(
                            fpath.read_bytes(), fpath.name, settings,
                            cek_ekstensi=False,  # dataset Orange memakai .bmp
                        )
                    )
                except Exception as e:
                    logger.warning("Gagal memproses %s: %s", fpath.name, e)
            if not tensors:
                continue
            feats = extractor.ekstrak(tensors)
            fitur_list.append(feats)
            labels_list.extend([label] * feats.shape[0])
            logger.info("   %s: %d/%d citra selesai", folder, min(i + BATCH, len(files)), len(files))

    if not fitur_list:
        raise ValueError(f"Tidak ada citra yang berhasil diekstrak di {data_dir}")

    X = np.vstack(fitur_list).astype(np.float32)
    y = np.array(labels_list, dtype=object)

    if output_features_dir:
        output_features_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_features_dir / "X_features.npy", X)
        np.save(output_features_dir / "y_labels.npy", y)
        logger.info("Fitur embedding disimpan di %s", output_features_dir)

    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Pelatihan & Evaluasi 5 Model Skema Orange (MLP, SVM, RF, kNN, NB)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Direktori dataset citra (subfolder: Normal_Sperm, Abnormal_Sperm, Non-Sperm)",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        help="Path ke berkas matriks fitur .npy (jika sudah diekstrak sebelumnya)",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        help="Path ke berkas label .npy",
    )
    parser.add_argument(
        "--init-baseline",
        action="store_true",
        help="Inisialisasi baseline model agar backend langsung siap dijalankan",
    )
    parser.add_argument(
        "--output-model-dir",
        type=str,
        default=str(BASE_DIR / "app" / "models"),
        help="Folder penyimpanan model .pkl",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Batasi jumlah citra per kelas (untuk uji cepat)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_model_dir)

    if args.data_dir:
        data_path = Path(args.data_dir)
        logger.info("Memulai ekstraksi fitur dan pelatihan dari %s...", data_path)
        X, y = load_dataset_from_images(data_path, max_per_class=args.max_per_class)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        evaluate_models(X_train, X_test, y_train, y_test)
        logger.info("Menyimpan model terlatih ke %s...", output_dir)
        train_and_save_all(output_dir, X, y)
        print(f"\n[SUKSES] Seluruh 5 model skema Orange berhasil dilatih dan disimpan di: {output_dir}")

    elif args.features_file and args.labels_file:
        logger.info("Memuat fitur dari %s dan label dari %s...", args.features_file, args.labels_file)
        X = np.load(args.features_file)
        y = np.load(args.labels_file)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        evaluate_models(X_train, X_test, y_train, y_test)
        logger.info("Menyimpan model terlatih ke %s...", output_dir)
        train_and_save_all(output_dir, X, y)
        print(f"\n[SUKSES] Seluruh 5 model skema Orange berhasil dilatih dan disimpan di: {output_dir}")

    else:
        logger.info("Menginisialisasi baseline model skema Orange ke %s...", output_dir)
        saved = train_and_save_all(output_dir)
        print("\n[SUKSES] Baseline 5 model skema Orange berhasil diinisialisasi:")
        for k, v in saved.items():
            print(f"  - {k.upper():<5}: {v.name}")


if __name__ == "__main__":
    main()
