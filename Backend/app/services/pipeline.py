"""Pipeline inferensi.

Menghimpun feature extractor (SqueezeNet) dan predictor (5 learner skema Orange)
menjadi satu alur yang dipakai layer HTTP.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import Settings
from app.schemas.prediction import (
    ModelComparisonItem,
    ModelComparisonResponse,
    PredictionResponse,
)
from app.services.feature_extractor import FeatureExtractor
from app.services.models_factory import MODEL_SPECS
from app.services.predictor import Predictor
from app.utils.errors import InferensiGagal
from app.utils.image_utils import buka_dan_preprocess

logger = logging.getLogger(__name__)


from app.services.trainer import TrainerService


class KlasifikasiService:
    """Pemilik pipeline: prapemrosesan -> SqueezeNet -> ML Learners -> respons."""

    def __init__(
        self,
        settings: Settings,
        extractor: Optional[FeatureExtractor] = None,
        predictor: Optional[Predictor] = None,
        trainer: Optional[TrainerService] = None,
    ) -> None:
        self._settings = settings
        self.extractor = extractor if extractor is not None else FeatureExtractor(settings)
        self.predictor = predictor if predictor is not None else Predictor(settings)
        self.trainer = trainer if trainer is not None else TrainerService(settings)

    def reload_models(self) -> None:
        """Muat ulang seluruh model dari direktori model ke memori."""
        self.predictor._muat()

    # ------------------------------------------------------------------
    @property
    def model_loaded(self) -> bool:
        """Siap melayani prediksi bila kedua komponen siap."""
        return self.extractor.siap and self.predictor.siap

    @property
    def active_model(self) -> str:
        return self.predictor.active_model

    @property
    def kelas(self) -> List[str]:
        return list(self.predictor.label)

    @property
    def device(self) -> str:
        return self.extractor.device

    @property
    def pesan_status(self) -> str:
        if self.model_loaded:
            return f"Model siap melayani prediksi (Aktif: {self.active_model.upper()})."
        bagian = []
        if not self.extractor.siap:
            bagian.append(f"SqueezeNet: {self.extractor.pesan}")
        if not self.predictor.siap:
            bagian.append(f"Predictor: {self.predictor.pesan}")
        return " | ".join(bagian)

    def daftar_model(self) -> List[Dict[str, Any]]:
        """Daftar model dari skema Orange."""
        return self.predictor.daftar_model()

    def set_active_model(self, model_id: str) -> bool:
        """Ubah model aktif bawaan di runtime."""
        return self.predictor.set_active_model(model_id)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def prediksi_satu(
        self,
        nama: str,
        data: bytes,
        model_name: Optional[str] = None,
    ) -> PredictionResponse:
        return self.prediksi_massal([(nama, data)], model_name=model_name)[0]

    def prediksi_massal(
        self,
        berkas: Sequence[Tuple[str, bytes]],
        model_name: Optional[str] = None,
    ) -> List[PredictionResponse]:
        """Proses `(nama, bytes)` dan kembalikan satu respons per berkas."""
        if not berkas:
            raise InferensiGagal("Tidak ada berkas untuk diproses.")

        tensor = [buka_dan_preprocess(data, nama, self._settings) for nama, data in berkas]

        mulai = time.perf_counter()
        fitur = self.extractor.ekstrak(tensor)
        hasil = self.predictor.prediksi(list(fitur), model_name=model_name)
        durasi = (time.perf_counter() - mulai) * 1000.0

        if len(hasil) != len(berkas):
            raise InferensiGagal(
                f"Jumlah hasil ({len(hasil)}) tidak cocok jumlah berkas ({len(berkas)})."
            )

        respons: List[PredictionResponse] = []
        for (nama, _), item in zip(berkas, hasil):
            m_used = str(item.get("model_id", model_name or self.active_model))
            respons.append(
                PredictionResponse(
                    filename=nama,
                    predicted_class=str(item["label"]),
                    confidence=_bulat(float(item["confidence"])),
                    probabilities={
                        str(k): _bulat(float(v))
                        for k, v in dict(item["probabilities"]).items()  # type: ignore[arg-type]
                    },
                    processing_time_ms=round(durasi / len(berkas), 2),
                    model_used=m_used,
                )
            )
        return respons

    def prediksi_komparasi(self, nama: str, data: bytes) -> ModelComparisonResponse:
        """Jalankan citra ke 5 model skema Orange dan tampilkan perbandingan (Orange Test & Score)."""
        tensor = [buka_dan_preprocess(data, nama, self._settings)]
        mulai = time.perf_counter()
        fitur = self.extractor.ekstrak(tensor)
        komparasi = self.predictor.prediksi_komparasi(list(fitur))
        durasi = (time.perf_counter() - mulai) * 1000.0

        items: List[ModelComparisonItem] = []
        mlp_pred = ""
        for m_id, rows in komparasi.items():
            if not rows:
                continue
            row = rows[0]
            name = MODEL_SPECS.get(m_id, {}).get("name", m_id.upper())
            lbl = str(row["label"])
            if m_id == "mlp":
                mlp_pred = lbl
            items.append(
                ModelComparisonItem(
                    model_id=m_id,
                    model_name=name,
                    predicted_class=lbl,
                    confidence=_bulat(float(row["confidence"])),
                    probabilities={
                        str(k): _bulat(float(v))
                        for k, v in dict(row["probabilities"]).items()  # type: ignore[arg-type]
                    },
                )
            )

        best = mlp_pred or (items[0].predicted_class if items else "Unknown")
        return ModelComparisonResponse(
            filename=nama,
            processing_time_ms=round(durasi, 2),
            best_prediction=best,
            comparisons=items,
        )

    # ------------------------------------------------------------------
    def ringkasan(self) -> Dict[str, object]:
        """Objek status untuk `GET /health`."""
        loaded_models = [m["id"] for m in self.daftar_model() if m["is_loaded"]]
        return {
            "model_loaded": self.model_loaded,
            "device": self.device,
            "classes": self.kelas,
            "active_model": self.active_model,
            "available_models": loaded_models,
            "message": self.pesan_status,
        }


def _bulat(nilai: float) -> float:
    """Batasi ke 6 angka desimal agar respons JSON ringkas."""
    return float(np.round(nilai, 6))