"""Konfigurasi aplikasi.

Nilai dibaca dari environment variable dengan nama huruf besar dan garis bawah,
misal `MODEL_DIR` atau `MAX_BATCH_FILES`. Bila variable tidak diset, dipakai
nilai bawaan di bawah.

Arsitektur mengikuti skema Orange Data Mining (`SMIDS.ows`):
    Import Images -> Image Embedding (SqueezeNet) -> 5 Learners -> Test & Score

Widget "Image Embedding" Orange memakai embedder **SqueezeNet** (lokal, ndf)
dengan:
    - ukuran input 227 x 227 piksel,
    - prapemrosesan Caffe: RGB -> BGR lalu dikurangi mean piksel
      `[104.006, 116.669, 122.679]` (urutan BGR),
    - embedding = lapisan pra-softmax `flatten10` -> vektor 1000 dimensi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, List, Sequence, Tuple

# app/ -> Backend/
BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _env_list(name: str, default: Sequence[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# --- Konstanta prapemrosesan SqueezeNet Orange (Caffe) -----------------------
# Ukuran input SqueezeNet pada widget Image Embedding Orange.
IMAGE_SIZE: Tuple[int, int] = (227, 227)

# Mean piksel ImageNet (urutan BGR) yang dikurangkan setelah RGB -> BGR.
CAFFE_MEAN_BGR: Tuple[float, float, float] = (104.006, 116.669, 122.679)

# Dimensi embedding (lapisan pra-softmax `flatten10` SqueezeNet).
EMBEDDING_DIM: int = 1000


@dataclass(frozen=True)
class Settings:
    """Snapshot konfigurasi yang tidak bisa diubah setelah dibuat."""

    app_name: str
    version: str

    # --- Server ---
    host: str
    port: int
    log_level: str
    # "*" mengizinkan semua origin (dibutuhkan integrasi mobile/web front-end)
    cors_origins: Tuple[str, ...]

    # --- Model ---
    model_dir: Path
    mlp_path: Path
    device: str              # selalu "cpu" (embedder SqueezeNet ndf berbasis numpy)
    mlp_pickle_name: str
    active_model: str        # "mlp" | "svm" | "rf" | "knn" | "nb" (5 learner skema Orange)
    auto_train_baseline: bool  # latih baseline jika berkas model belum tersedia

    # --- Prapemrosesan citra (SqueezeNet Orange) ---
    image_size: Tuple[int, int]          # (227, 227) untuk PIL.resize
    caffe_mean_bgr: Tuple[float, float, float]

    # --- Validasi unggahan ---
    allowed_extensions: FrozenSet[str]
    allowed_mime: FrozenSet[str]
    max_pixels: int                      # guard decompression bomb
    max_upload_mb: int

    # --- Batas batch ---
    max_batch_files: int

    @property
    def device_resolved(self) -> str:
        # Embedder SqueezeNet (ndf) berjalan murni di numpy -> CPU.
        return "cpu"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    model_dir = Path(_env("MODEL_DIR", str(BASE_DIR / "app" / "models")))
    mlp_name = _env("MLP_PICKLE_NAME", "mlp_model.pkl")

    return Settings(
        app_name=_env("APP_NAME", "CAVISA - Sperm Abnormality Classification API"),
        version=_env("APP_VERSION", "1.0.0"),
        host=_env("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8000),
        log_level=_env("LOG_LEVEL", "info"),
        cors_origins=tuple(_env_list("CORS_ORIGINS", ["*"])),
        model_dir=model_dir,
        mlp_path=model_dir / mlp_name,
        device=_env("DEVICE", "cpu"),
        mlp_pickle_name=mlp_name,
        active_model=_env("ACTIVE_MODEL", "mlp"),
        auto_train_baseline=_env_bool("AUTO_TRAIN_BASELINE", True),
        image_size=IMAGE_SIZE,
        caffe_mean_bgr=CAFFE_MEAN_BGR,
        allowed_extensions=frozenset({".jpg", ".jpeg", ".png", ".bmp"}),
        allowed_mime=frozenset({"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/x-ms-bmp"}),
        max_pixels=_env_int("MAX_PIXELS", 40_000_000),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 10),
        max_batch_files=_env_int("MAX_BATCH_FILES", 20),
    )


# Label kelas kanonik (yang dikembalikan API).
CLASS_LABELS: Tuple[str, ...] = ("Normal", "Abnormal", "Non-sperm")

# Pemetaan nama folder dataset (kelas Orange `Normal_Sperm`, `Abnormal_Sperm`,
# `Non-Sperm`) ke label kanonik backend.
DATASET_FOLDER_TO_LABEL = {
    "Normal_Sperm": "Normal",
    "Abnormal_Sperm": "Abnormal",
    "Non-Sperm": "Non-sperm",
    "normal_sperm": "Normal",
    "abnormal_sperm": "Abnormal",
    "non_sperm": "Non-sperm",
    "Normal": "Normal",
    "Abnormal": "Abnormal",
    "Non-sperm": "Non-sperm",
}

_CANONICAL = {
    "normal": "Normal",
    "normal_sperm": "Normal",
    "abnormal": "Abnormal",
    "abnormal_sperm": "Abnormal",
    "non-sperm": "Non-sperm",
    "nonsperm": "Non-sperm",
    "non sperm": "Non-sperm",
    "non_sperm": "Non-sperm",
}


def canonical_label(raw: object) -> str:
    """Normalkan nama kelas (termasuk folder Orange) ke label kanonik."""
    key = str(raw).strip().lower()
    if key in _CANONICAL:
        return _CANONICAL[key]
    # label tak dikenal dikembalikan apa adanya supaya informasinya tidak hilang
    return str(raw)
