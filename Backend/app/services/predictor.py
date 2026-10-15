"""Klasifikasi dengan model-model Machine Learning dari Skema Orange.

Mendukung 5 learner sesuai diagram Orange Data Mining & paper:
  - Neural Network / MLP (Model Utama, F1: 0.854)
  - Support Vector Machine (SVM)
  - Random Forest (RF)
  - k-Nearest Neighbors (kNN)
  - Naive Bayes (NB)

Model dimuat dari `app/models/`. Bila belum tersedia, server dapat
menginisialisasi baseline secara otomatis agar langsung siap melayani request.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import CLASS_LABELS, Settings, canonical_label
from app.services.models_factory import MODEL_SPECS, normalize_model_id, train_and_save_all
from app.utils.errors import InferensiGagal, ModelBelumSiap

logger = logging.getLogger(__name__)


class Predictor:
    """Pembungkus model-model klasifikasi scikit-learn dari skema Orange."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._models: Dict[str, Any] = {}
        self._labels: Dict[str, Tuple[str, ...]] = {}
        self._primary_id: str = "mlp"
        self._siap: bool = False
        self._pesan: str = "Model belum dimuat."
        self._muat()

    # ------------------------------------------------------------------
    # Pemuatan
    # ------------------------------------------------------------------
    def _muat_satu_model(self, file_path) -> Optional[Tuple[Any, Tuple[str, ...]]]:
        if not file_path.exists():
            return None
        try:
            import joblib

            model = joblib.load(file_path)
            if not hasattr(model, "predict_proba"):
                logger.warning("Berkas '%s' tidak memiliki predict_proba.", file_path.name)
                return None
            mentah = getattr(model, "classes_", None)
            if mentah is None or len(mentah) == 0:
                logger.warning("Berkas '%s' tidak memiliki classes_.", file_path.name)
                return None
            labels = tuple(canonical_label(c) for c in mentah)
            return model, labels
        except Exception as exc:
            logger.error("Gagal memuat '%s': %s", file_path.name, exc)
            return None

    def _muat(self) -> None:
        model_dir = self._settings.model_dir
        primary_req = normalize_model_id(getattr(self._settings, "active_model", "mlp"))
        self._primary_id = primary_req if primary_req in MODEL_SPECS else "mlp"

        # Cek apakah berkas model utama ada (misal mlp_model.pkl atau best_mlp_sperm_model.pkl)
        mlp_path = self._settings.mlp_path
        alt_mlp_path = model_dir / "best_mlp_sperm_model.pkl"
        has_primary = mlp_path.exists() or alt_mlp_path.exists()

        # Inisialisasi otomatis jika belum ada dan auto_train_baseline aktif
        if not has_primary and getattr(self._settings, "auto_train_baseline", True):
            logger.info("Model belum ditemukan di '%s'. Melakukan inisialisasi baseline model Orange...", model_dir)
            try:
                train_and_save_all(model_dir)
            except Exception as exc:
                logger.warning("Gagal inisialisasi baseline model: %s", exc)

        # Muat semua model Orange yang ada di folder models
        for m_id, spec in MODEL_SPECS.items():
            path = model_dir / spec["file"]
            if not path.exists() and spec.get("alt_file"):
                path = model_dir / spec["alt_file"]

            loaded = self._muat_satu_model(path)
            if loaded:
                model, labels = loaded
                self._models[m_id] = model
                self._labels[m_id] = labels
                logger.info(
                    "Model %s ('%s') siap. Fitur: %s. Kelas: %s.",
                    m_id.upper(),
                    path.name,
                    getattr(model, "n_features_in_", "?"),
                    ", ".join(labels),
                )

        # Tentukan status kesiapan model utama
        if self._primary_id in self._models:
            self._siap = True
            self._pesan = f"Model {MODEL_SPECS[self._primary_id]['name']} siap."
        elif "mlp" in self._models:
            self._primary_id = "mlp"
            self._siap = True
            self._pesan = "Model Neural Network (MLP) siap."
        elif self._models:
            self._primary_id = next(iter(self._models.keys()))
            self._siap = True
            self._pesan = f"Model {MODEL_SPECS[self._primary_id]['name']} siap."
        else:
            self._siap = False
            self._pesan = f"Tidak ada model yang termuat di '{model_dir}'."
            logger.warning(self._pesan)

    # ------------------------------------------------------------------
    # Properti
    # ------------------------------------------------------------------
    @property
    def siap(self) -> bool:
        return self._siap

    @property
    def pesan(self) -> str:
        return self._pesan

    @property
    def active_model(self) -> str:
        return self._primary_id

    def set_active_model(self, model_id: str) -> bool:
        norm_id = normalize_model_id(model_id)
        if norm_id in self._models:
            self._primary_id = norm_id
            self._pesan = f"Model {MODEL_SPECS[norm_id]['name']} siap (Aktif)."
            return True
        return False

    @property
    def label(self) -> Tuple[str, ...]:
        """Label kelas untuk model aktif."""
        return self._labels.get(self._primary_id, CLASS_LABELS)

    @property
    def n_features(self) -> Optional[int]:
        model = self._models.get(self._primary_id)
        if model is None:
            return None
        nilai = getattr(model, "n_features_in_", None)
        return int(nilai) if nilai is not None else None

    def daftar_model(self) -> List[Dict[str, Any]]:
        """Daftar 5 model Orange beserta status kesiapannya."""
        daftar = []
        for m_id, spec in MODEL_SPECS.items():
            loaded = m_id in self._models
            daftar.append(
                {
                    "id": m_id,
                    "name": spec["name"],
                    "description": spec["description"],
                    "paper_table": spec["paper_table"],
                    "is_active": m_id == self._primary_id,
                    "is_loaded": loaded,
                }
            )
        return daftar

    def _wajib_siap(self, model_id: Optional[str] = None) -> Tuple[Any, Tuple[str, ...]]:
        target_id = normalize_model_id(model_id) if model_id else self._primary_id
        if target_id not in self._models:
            raise ModelBelumSiap(
                f"Model '{target_id}' belum siap. {self._pesan} "
                f"Model yang tersedia: {list(self._models.keys())}."
            )
        return self._models[target_id], self._labels[target_id]

    # ------------------------------------------------------------------
    # Inferensi
    # ------------------------------------------------------------------
    def prediksi(
        self,
        features: Sequence[np.ndarray],
        model_name: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Klasifikasikan batch fitur 1000-dim dengan model yang dipilih."""
        model_id = normalize_model_id(model_name) if model_name else self._primary_id
        model, labels = self._wajib_siap(model_id)

        daftar_fitur = list(features)
        if not daftar_fitur:
            raise InferensiGagal("Batch fitur kosong.")

        matriks = np.asarray(daftar_fitur, dtype=np.float32)
        if matriks.ndim != 2:
            raise InferensiGagal(f"Forma fitur tidak diharapkan: {matriks.shape}.")

        n_in = getattr(model, "n_features_in_", None)
        if n_in and matriks.shape[1] != n_in:
            raise InferensiGagal(
                f"Dimensi fitur {matriks.shape[1]} tidak cocok dengan model "
                f"({n_in}). Model dilatih pada vektor SqueezeNet Orange 1000-dim "
                f"(embedder ndf, pra-softmax flatten10)."
            )

        try:
            with self._lock:
                proba = model.predict_proba(matriks)
        except Exception as exc:
            raise InferensiGagal(f"Gagal menjalankan inferensi model '{model_id}': {exc}") from exc

        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim != 2 or proba.shape[1] != len(labels):
            raise InferensiGagal(
                f"Jumlah kolom probabilitas tidak cocok dengan jumlah label "
                f"({len(labels)})."
            )

        hasil: List[Dict[str, object]] = []
        for baris in proba:
            indeks = int(np.argmax(baris))
            hasil.append(
                {
                    "model_id": model_id,
                    "label": labels[indeks],
                    "confidence": float(baris[indeks]),
                    "probabilities": {
                        lbl: float(nilai) for lbl, nilai in zip(labels, baris)
                    },
                }
            )
        return hasil

    def prediksi_komparasi(
        self,
        features: Sequence[np.ndarray],
    ) -> Dict[str, List[Dict[str, object]]]:
        """Jalankan inferensi ke seluruh 5 model Orange yang termuat (seperti Test & Score)."""
        komparasi: Dict[str, List[Dict[str, object]]] = {}
        for m_id in self._models:
            komparasi[m_id] = self.prediksi(features, model_name=m_id)
        return komparasi