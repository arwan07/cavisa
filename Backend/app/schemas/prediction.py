"""Skema Pydantic v2 untuk request dan response."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Respons `GET /health`."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "service": "Sperm Abnormality Classification API",
                "version": "1.0.0",
                "model_loaded": True,
                "device": "cpu",
                "active_model": "mlp",
                "available_models": ["mlp", "svm", "rf", "knn", "nb"],
                "classes": ["Normal", "Abnormal", "Non-sperm"],
            }
        }
    )

    status: str = Field(..., examples=["ok"])
    service: str
    version: str
    model_loaded: bool
    device: str
    classes: List[str]
    active_model: Optional[str] = "mlp"
    available_models: Optional[List[str]] = Field(default_factory=list)
    message: Optional[str] = None


class PredictionResponse(BaseModel):
    """Respons `POST /api/v1/predict` dan tiap elemen pada `predict-batch`."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "sperma-01.jpg",
                "predicted_class": "Normal",
                "confidence": 0.9421,
                "model_used": "mlp",
                "probabilities": {
                    "Normal": 0.9421,
                    "Abnormal": 0.0512,
                    "Non-sperm": 0.0067,
                },
                "processing_time_ms": 87.4,
            }
        }
    )

    filename: str = Field(..., description="Nama berkas yang diproses.")
    predicted_class: str = Field(
        ...,
        description="Kelas hasil prediksi: Normal, Abnormal, atau Non-sperm.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilitas tertinggi dari kelas terpilih.",
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilitas seluruh kelas, nilainya berjumlah 1.0.",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Waktu proses dalam milidetik.",
    )
    model_used: Optional[str] = Field(
        default="mlp",
        description="Algoritma model yang digunakan (mlp, svm, rf, knn, nb).",
    )


class ModelInfo(BaseModel):
    """Metadata untuk satu learner dari skema Orange."""

    id: str = Field(..., description="ID model: mlp, svm, rf, knn, nb")
    name: str = Field(..., description="Nama resmi model")
    description: str = Field(..., description="Deskripsi hiperparameter & performa")
    paper_table: str = Field(..., description="Tabel referensi di paper")
    is_active: bool = Field(..., description="Apakah model ini yang sedang aktif")
    is_loaded: bool = Field(..., description="Apakah model sudah siap di memori")


class ModelListResponse(BaseModel):
    """Respons `GET /api/v1/models`."""

    active_model: str
    models: List[ModelInfo]


class ModelComparisonItem(BaseModel):
    """Hasil prediksi dari satu model dalam mode komparasi."""

    model_id: str
    model_name: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


class ModelComparisonResponse(BaseModel):
    """Respons `POST /api/v1/predict-compare` (seperti widget Test and Score di Orange)."""

    filename: str
    processing_time_ms: float
    best_prediction: str
    comparisons: List[ModelComparisonItem]


class LegacyPredictResponse(BaseModel):
    """Format respons kompatibel dengan `sperm_classification_api.py` dari Gemini."""

    status: str = "success"
    prediction: str
    confidence: str
    embedding_size: int = 1000
    model_used: Optional[str] = "mlp"


class ErrorResponse(BaseModel):
    """Bentuk galat agar klien bisa menampilkan `detail` apa adanya."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Format berkas '.txt' tidak didukung. Gunakan: .jpeg, .jpg, .png.",
                "code": "invalid_image",
            }
        }
    )

    detail: str
    code: Optional[str] = None


class DatasetFolderInfo(BaseModel):
    """Informasi folder dataset yang tersedia di sistem."""

    id: str = Field(..., description="ID folder: data_asli, data_latih, data_validasi, data_uji")
    name: str = Field(..., description="Nama tampilan folder")
    path: str = Field(..., description="Path absolut atau relatif folder")
    exists: bool = Field(..., description="Apakah folder ada di sistem")
    total_images: int = Field(0, description="Total jumlah gambar dalam dataset")
    counts: Dict[str, int] = Field(default_factory=dict, description="Jumlah gambar per kelas")


class DatasetListResponse(BaseModel):
    """Respons `GET /api/v1/datasets`."""

    datasets: List[DatasetFolderInfo]


class TrainRequest(BaseModel):
    """Payload `POST /api/v1/train`."""

    dataset_folder: str = Field(
        default="data_latih",
        description="Pilihan folder dataset: data_latih, data_asli, data_validasi, data_uji, atau path kustom",
    )
    max_per_class: Optional[int] = Field(
        default=None,
        description="Batasan jumlah citra per kelas (None = seluruh data, angka = uji cepat)",
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="Daftar model yang dilatih (None = ke-5 model Orange: mlp, svm, rf, knn, nb)",
    )
    test_size: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Proporsi evaluasi test-split",
    )


class TrainModelMetric(BaseModel):
    """Metrik hasil evaluasi per model."""

    model_id: str
    name: str
    paper_table: str
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    confusion_matrix: Dict[str, List[int]]
    classes: List[str]


class TrainStatusResponse(BaseModel):
    """Respons status pelatihan `GET /api/v1/train/status`."""

    is_training: bool = Field(..., description="Apakah proses pelatihan sedang berlangsung")
    progress_percent: int = Field(0, description="Persentase progres (0-100)")
    current_stage: str = Field("idle", description="Tahapan yang sedang berjalan")
    dataset_used: Optional[str] = None
    total_images: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    results: Optional[Dict[str, TrainModelMetric]] = None
    saved_models: List[str] = Field(default_factory=list)