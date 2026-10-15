"""Layanan Pelatihan Model (Trainer Service).

Memungkinkan pelatihan ke-5 model Machine Learning skema Orange
langsung melalui antarmuka API / Frontend pada folder dataset tertentu
(misal: data_latih, data_asli, data_validasi, data_uji).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.config import BASE_DIR, DATASET_FOLDER_TO_LABEL, Settings, get_settings
from app.schemas.prediction import (
    DatasetFolderInfo,
    TrainModelMetric,
    TrainRequest,
    TrainStatusResponse,
)
from app.services.feature_extractor import FeatureExtractor
from app.services.models_factory import CLASS_NAMES, MODEL_SPECS, normalize_model_id, train_and_save_all
from app.utils.errors import InferensiGagal
from app.utils.image_utils import buka_dan_preprocess

logger = logging.getLogger(__name__)

# Direktori root Nalatico (induk dari folder Backend/)
PROJECT_ROOT = BASE_DIR.parent

DATASET_FOLDERS = ["Normal_Sperm", "Abnormal_Sperm", "Non-Sperm"]
EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png")


class TrainerService:
    """Mengelola proses pelatihan model asinkron dengan pelacakan progres."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None

        self._is_training: bool = False
        self._progress_percent: int = 0
        self._current_stage: str = "idle"
        self._dataset_used: Optional[str] = None
        self._total_images: int = 0
        self._start_time: float = 0.0
        self._elapsed_seconds: float = 0.0
        self._error: Optional[str] = None
        self._results: Optional[Dict[str, TrainModelMetric]] = None
        self._saved_models: List[str] = []

    # ------------------------------------------------------------------
    # Dataset Discovery
    # ------------------------------------------------------------------
    def list_datasets(self) -> List[DatasetFolderInfo]:
        """Daftar folder dataset standar yang ada di proyek Nalatico."""
        known_folders = [
            ("data_latih", "Data Latih (80% - Pelatihan Model)"),
            ("data_validasi", "Data Validasi (10% - Validasi)"),
            ("data_uji", "Data Uji (10% - Evaluasi/Test)"),
            ("data_asli", "Data Asli (Master Dataset SMIDS)"),
        ]

        hasil: List[DatasetFolderInfo] = []
        for folder_id, label_name in known_folders:
            folder_path = self.resolve_dataset_path(folder_id)
            exists = folder_path.exists() and folder_path.is_dir()
            counts: Dict[str, int] = {}
            total = 0

            if exists:
                for sub in folder_path.iterdir():
                    if sub.is_dir() and not sub.is_symlink() and not sub.name.startswith("."):
                        c = len([f for f in sub.iterdir() if f.is_file() and not f.name.startswith(".")])
                        counts[sub.name] = c
                        total += c

            hasil.append(
                DatasetFolderInfo(
                    id=folder_id,
                    name=label_name,
                    path=str(folder_path),
                    exists=exists,
                    total_images=total,
                    counts=counts,
                )
            )
        return hasil

    def resolve_dataset_path(self, folder_str: str) -> Path:
        """Resolusi nama folder atau path dataset."""
        p = Path(folder_str)
        if p.is_absolute() and p.exists():
            return p

        candidates = [
            PROJECT_ROOT / folder_str,
            BASE_DIR / folder_str,
            Path(folder_str).resolve(),
        ]
        for c in candidates:
            if c.exists():
                return c

        return PROJECT_ROOT / folder_str

    # ------------------------------------------------------------------
    # Status Pelatihan
    # ------------------------------------------------------------------
    def get_status(self) -> TrainStatusResponse:
        """Ambil status proses pelatihan saat ini."""
        with self._lock:
            elapsed = time.time() - self._start_time if self._is_training else self._elapsed_seconds
            return TrainStatusResponse(
                is_training=self._is_training,
                progress_percent=self._progress_percent,
                current_stage=self._current_stage,
                dataset_used=self._dataset_used,
                total_images=self._total_images,
                elapsed_seconds=round(elapsed, 1),
                error=self._error,
                results=self._results,
                saved_models=self._saved_models,
            )

    # ------------------------------------------------------------------
    # Memulai Pelatihan
    # ------------------------------------------------------------------
    def start_training(
        self,
        req: TrainRequest,
        on_success_reload_fn: Optional[Callable[[], None]] = None,
    ) -> TrainStatusResponse:
        """Jalankan proses pelatihan di background thread."""
        with self._lock:
            if self._is_training:
                raise InferensiGagal("Pelatihan sedang berjalan. Harap tunggu hingga selesai.")

            folder_path = self.resolve_dataset_path(req.dataset_folder)
            if not folder_path.exists() or not folder_path.is_dir():
                raise InferensiGagal(f"Folder dataset '{req.dataset_folder}' tidak ditemukan di {folder_path}.")

            self._is_training = True
            self._progress_percent = 0
            self._current_stage = f"Mempersiapkan dataset dari {folder_path.name}…"
            self._dataset_used = folder_path.name
            self._total_images = 0
            self._start_time = time.time()
            self._elapsed_seconds = 0.0
            self._error = None
            self._results = None
            self._saved_models = []

            self._thread = threading.Thread(
                target=self._run_training_job,
                args=(folder_path, req, on_success_reload_fn),
                daemon=True,
            )
            self._thread.start()

            return self.get_status()

    # ------------------------------------------------------------------
    # Job Pelatihan di Background Thread
    # ------------------------------------------------------------------
    def _run_training_job(
        self,
        folder_path: Path,
        req: TrainRequest,
        on_success_reload_fn: Optional[Callable[[], None]],
    ) -> None:
        try:
            logger.info("Memulai proses pelatihan dari folder: %s", folder_path)
            extractor = FeatureExtractor(self._settings)
            if not extractor.siap:
                raise RuntimeError(f"SqueezeNet feature extractor gagal dimuat: {extractor.pesan}")

            # 1. Kumpulkan seluruh berkas citra
            with self._lock:
                self._current_stage = "Memindai file citra..."
                self._progress_percent = 5

            tasks: List[Tuple[Path, str]] = []
            for folder in DATASET_FOLDERS:
                kelas_dir = folder_path / folder
                if not kelas_dir.exists():
                    alt_folder = folder.lower().replace("-", "_")
                    if (folder_path / alt_folder).exists():
                        kelas_dir = folder_path / alt_folder
                        folder = alt_folder
                    else:
                        continue

                label = DATASET_FOLDER_TO_LABEL.get(folder, folder)
                files = [
                    p for p in sorted(kelas_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.name.startswith(".")
                ]
                if req.max_per_class:
                    files = files[:req.max_per_class]

                for f in files:
                    tasks.append((f, label))

            if not tasks:
                raise ValueError(f"Tidak ada file citra yang ditemukan di folder {folder_path}")

            total_all = len(tasks)
            with self._lock:
                self._total_images = total_all
                self._current_stage = f"Mengekstrak fitur SqueezeNet dari {total_all} citra…"
                self._progress_percent = 10

            # 2. Ekstraksi Fitur (Batch)
            BATCH_SIZE = 64
            all_features = []
            all_labels = []

            for i in range(0, total_all, BATCH_SIZE):
                chunk = tasks[i : i + BATCH_SIZE]
                tensors = []
                chunk_labels = []

                for fpath, label in chunk:
                    try:
                        t = buka_dan_preprocess(
                            fpath.read_bytes(),
                            fpath.name,
                            self._settings,
                            cek_ekstensi=False,
                        )
                        tensors.append(t)
                        chunk_labels.append(label)
                    except Exception as exc:
                        logger.warning("Gagal memproses %s: %s", fpath.name, exc)

                if tensors:
                    feats = extractor.ekstrak(tensors)
                    all_features.append(feats)
                    all_labels.extend(chunk_labels)

                # Update progress ekstraksi fitur (10% - 70%)
                curr_done = min(i + BATCH_SIZE, total_all)
                p_val = 10 + int((curr_done / total_all) * 60)
                with self._lock:
                    self._progress_percent = p_val
                    self._current_stage = f"Ekstraksi fitur: {curr_done}/{total_all} citra ({p_val}%)"

            if not all_features:
                raise ValueError("Ekstraksi fitur tidak menghasilkan data.")

            X = np.vstack(all_features).astype(np.float32)
            y = np.array(all_labels, dtype=object)

            # 3. Train Test Split
            with self._lock:
                self._progress_percent = 75
                self._current_stage = "Membagi dataset (train/test split)..."

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=req.test_size, random_state=42, stratify=y
            )

            # 4. Pelatihan & Evaluasi Model
            with self._lock:
                self._progress_percent = 80
                self._current_stage = "Melatih dan mengevaluasi 5 model skema Orange…"

            models_to_run = req.models if req.models else list(MODEL_SPECS.keys())
            metrics_dict: Dict[str, TrainModelMetric] = {}

            for m_id in models_to_run:
                norm_id = normalize_model_id(m_id)
                if norm_id not in MODEL_SPECS:
                    continue
                spec = MODEL_SPECS[norm_id]
                clf = spec["builder"]()
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                acc = float(np.mean(y_pred == y_test))
                f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                p_m = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
                r_m = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

                cm = confusion_matrix(y_test, y_pred, labels=list(CLASS_NAMES))
                cm_dict = {
                    CLASS_NAMES[row_idx]: [int(val) for val in row]
                    for row_idx, row in enumerate(cm)
                }

                metrics_dict[norm_id] = TrainModelMetric(
                    model_id=norm_id,
                    name=spec["name"],
                    paper_table=spec["paper_table"],
                    accuracy=round(acc, 4),
                    f1_macro=round(f1_m, 4),
                    precision_macro=round(p_m, 4),
                    recall_macro=round(r_m, 4),
                    confusion_matrix=cm_dict,
                    classes=list(CLASS_NAMES),
                )

            # 5. Latih Ulang dengan Keseluruhan Data & Simpan Model
            with self._lock:
                self._progress_percent = 90
                self._current_stage = "Menyimpan model terlatih ke disk…"

            saved_paths = train_and_save_all(self._settings.model_dir, X, y)
            saved_names = [p.name for p in saved_paths.values()]

            # 6. Reload model di server aktif jika ada callback
            if on_success_reload_fn:
                try:
                    on_success_reload_fn()
                    logger.info("Model aktif pada server berhasil diperbarui.")
                except Exception as exc:
                    logger.warning("Gagal memuat ulang model aktif: %s", exc)

            with self._lock:
                self._is_training = False
                self._progress_percent = 100
                self._current_stage = "Pelatihan selesai! Seluruh model siap digunakan."
                self._elapsed_seconds = round(time.time() - self._start_time, 1)
                self._results = metrics_dict
                self._saved_models = saved_names

            logger.info("Pelatihan selesai dalam %.1f detik.", self._elapsed_seconds)

        except Exception as exc:
            logger.error("Pelatihan gagal: %s", exc, exc_info=True)
            with self._lock:
                self._is_training = False
                self._error = str(exc)
                self._current_stage = f"Pelatihan gagal: {exc}"
                self._elapsed_seconds = round(time.time() - self._start_time, 1)
