"""Entrypoint FastAPI.

Menyediakan:
    GET  /health                -> status server & kesiapan model
    GET  /api/v1/models         -> daftar 5 learner skema Orange (MLP, SVM, RF, kNN, NB)
    POST /api/v1/predict        -> klasifikasi satu citra (field: file, param: model opsional)
    POST /api/v1/predict-batch  -> klasifikasi banyak citra (field: files, param: model opsional)
    POST /api/v1/predict-compare-> klasifikasi komparasi ke 5 model (Orange Test & Score)
    POST /predict               -> alias kompatibilitas penuh dengan sperm_classification_api.py
    GET  /                      -> ringkasan service & rute
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas.prediction import (
    DatasetListResponse,
    ErrorResponse,
    HealthResponse,
    LegacyPredictResponse,
    ModelComparisonResponse,
    ModelInfo,
    ModelListResponse,
    PredictionResponse,
    TrainRequest,
    TrainStatusResponse,
)
from app.services.pipeline import KlasifikasiService
from app.utils.errors import AppError, CitraTidakValid, FileKosong, ModelBelumSiap

logger = logging.getLogger(__name__)
settings = get_settings()

# Diisi sekali oleh lifespan saat start dan dibersihkan saat shutdown.
layanan: Optional[KlasifikasiService] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pramuat SqueezeNet dan Model Skema Orange sekali saja saat server start."""
    global layanan
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logger.info("Memuat pipeline model (perangkat '%s')…", settings.device_resolved)
    layanan = KlasifikasiService(settings)

    if layanan.model_loaded:
        logger.info(
            "Model siap pada '%s'. Model aktif: %s. Kelas: %s",
            layanan.device,
            layanan.active_model.upper(),
            ", ".join(layanan.kelas),
        )
    else:
        logger.warning("Server start tanpa model siap: %s", layanan.pesan_status)

    yield

    logger.info("Server dihentikan.")
    layanan = None


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "API klasifikasi morfologi sel sperma berbasis skema Orange Data Mining. "
        "Pipeline: SqueezeNet Orange/ndf Image Embedding (1000-dim) -> 5 ML Learners "
        "(MLP / SVM / Random Forest / kNN / Naive Bayes) -> Normal / Abnormal / Non-sperm."
    ),
    lifespan=lifespan,
)

# CORS untuk integrasi front-end/mobile.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Handler galat terpusat
# ---------------------------------------------------------------------------
@app.exception_handler(AppError)
async def handler_app_error(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.kode},
    )


# ---------------------------------------------------------------------------
# Endpoint Informasi & Kesehatan
# ---------------------------------------------------------------------------
@app.get("/", tags=["info"], summary="Informasi singkat service")
async def root() -> Dict[str, object]:
    return {
        "service": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /api/v1/predict",
            "predict_batch": "POST /api/v1/predict-batch",
            "predict_compare": "POST /api/v1/predict-compare",
            "models": "GET /api/v1/models",
            "legacy_predict": "POST /predict",
        },
        "active_model": layanan.active_model if layanan else settings.active_model,
        "model_loaded": bool(layanan and layanan.model_loaded),
    }


@app.get(
    "/health",
    tags=["info"],
    response_model=HealthResponse,
    summary="Status server dan kesiapan model",
)
async def health() -> HealthResponse:
    status_info = layanan.ringkasan() if layanan else {
        "model_loaded": False,
        "device": settings.device_resolved,
        "classes": [],
        "active_model": settings.active_model,
        "available_models": [],
        "message": "Server sedang menjalankan proses startup.",
    }
    return HealthResponse(
        status="ok" if layanan and layanan.model_loaded else "degraded",
        service=settings.app_name,
        version=settings.version,
        **status_info,  # type: ignore[arg-type]
    )


@app.get(
    "/api/v1/models",
    tags=["model"],
    response_model=ModelListResponse,
    summary="Daftar model algoritma learner dari skema Orange",
)
async def list_models() -> ModelListResponse:
    if layanan is None:
        raise ModelBelumSiap("Server belum selesai memuat model.")
    items = [
        ModelInfo(
            id=str(m["id"]),
            name=str(m["name"]),
            description=str(m["description"]),
            paper_table=str(m["paper_table"]),
            is_active=bool(m["is_active"]),
            is_loaded=bool(m["is_loaded"]),
        )
        for m in layanan.daftar_model()
    ]
    return ModelListResponse(active_model=layanan.active_model, models=items)


@app.post(
    "/api/v1/models/active",
    tags=["model"],
    summary="Ubah model aktif bawaan di server",
)
async def set_active_model(
    model: str = Query(..., description="ID model: mlp, svm, rf, knn, nb")
) -> Dict[str, Any]:
    if layanan is None:
        raise ModelBelumSiap("Server sedang memulai...")
    ok = layanan.set_active_model(model)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' tidak ditemukan atau gagal diaktifkan.",
        )
    return {"status": "ok", "active_model": layanan.active_model}


@app.post(
    "/api/v1/models/reload",
    tags=["model"],
    summary="Muat ulang seluruh file model dari disk ke memori",
)
async def reload_models() -> Dict[str, Any]:
    if layanan is None:
        raise ModelBelumSiap("Server sedang memulai...")
    layanan.reload_models()
    return {
        "status": "ok",
        "message": "Model berhasil dimuat ulang dari disk.",
        "active_model": layanan.active_model,
    }


# ---------------------------------------------------------------------------
# Endpoint Manajemen Dataset & Pelatihan Model
# ---------------------------------------------------------------------------
@app.get(
    "/api/v1/datasets",
    tags=["pelatihan"],
    response_model=DatasetListResponse,
    summary="Daftar dataset yang tersedia (data_asli, data_latih, data_validasi, data_uji)",
)
async def list_datasets() -> DatasetListResponse:
    if layanan is None:
        raise ModelBelumSiap("Server sedang memulai...")
    items = layanan.trainer.list_datasets()
    return DatasetListResponse(datasets=items)


@app.post(
    "/api/v1/train",
    tags=["pelatihan"],
    response_model=TrainStatusResponse,
    summary="Mulai proses pelatihan model Machine Learning dari folder tertentu",
)
async def start_training(req: TrainRequest) -> TrainStatusResponse:
    if layanan is None:
        raise ModelBelumSiap("Server belum siap.")
    return layanan.trainer.start_training(req, on_success_reload_fn=layanan.reload_models)


@app.get(
    "/api/v1/train/status",
    tags=["pelatihan"],
    response_model=TrainStatusResponse,
    summary="Status proses pelatihan saat ini beserta riwayat metrik evaluasi",
)
async def train_status() -> TrainStatusResponse:
    if layanan is None:
        raise ModelBelumSiap("Server belum siap.")
    return layanan.trainer.get_status()


# ---------------------------------------------------------------------------
# Helper pembacaan berkas
# ---------------------------------------------------------------------------
async def _baca(berkas: UploadFile, maks_bytes: int) -> Tuple[str, bytes]:
    """Baca dan validasi satu berkas unggahan."""
    nama = berkas.filename or "tanpa_nama"
    data = await berkas.read()
    if not data:
        raise FileKosong(f"Berkas '{nama}' kosong (0 byte).")
    if len(data) > maks_bytes:
        raise CitraTidakValid(
            f"Ukuran '{nama}' melebihi batas {maks_bytes // (1024*1024)} MB."
        )
    return nama, data


# ---------------------------------------------------------------------------
# Endpoint Prediksi
# ---------------------------------------------------------------------------
@app.post(
    "/api/v1/predict",
    tags=["prediksi"],
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Berkas citra tidak valid"},
        503: {"model": ErrorResponse, "description": "Model belum termuat"},
    },
    summary="Klasifikasi satu citra sperma (opsional: pilih model Orange)",
)
async def predict(
    file: UploadFile = File(..., description="Citra sperma (.jpg, .jpeg, .png, .bmp)"),
    model: Optional[str] = Query(
        None,
        description="Pilihan model learner: mlp, svm, rf, knn, nb (bawaan: aktif)",
    ),
) -> PredictionResponse:
    if layanan is None:
        raise ModelBelumSiap("Server belum selesai memuat model.")
    nama, data = await _baca(file, settings.max_upload_bytes)
    return layanan.prediksi_satu(nama, data, model_name=model)


@app.post(
    "/api/v1/predict-batch",
    tags=["prediksi"],
    response_model=List[PredictionResponse],
    responses={
        400: {"model": ErrorResponse, "description": "Salah satu berkas tidak valid"},
        503: {"model": ErrorResponse, "description": "Model belum termuat"},
    },
    summary="Klasifikasi banyak citra sekaligus (opsional: pilih model Orange)",
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="Daftar citra sperma (.jpg, .jpeg, .png, .bmp)"),
    model: Optional[str] = Query(
        None,
        description="Pilihan model learner: mlp, svm, rf, knn, nb (bawaan: aktif)",
    ),
) -> List[PredictionResponse]:
    if layanan is None:
        raise ModelBelumSiap("Server belum selesai memuat model.")
    if not files:
        raise CitraTidakValid("Tidak ada berkas yang dikirim.")
    if len(files) > settings.max_batch_files:
        raise CitraTidakValid(
            f"Jumlah berkas ({len(files)}) melebihi batas "
            f"{settings.max_batch_files}."
        )
    maks = settings.max_upload_bytes
    berkas = [await _baca(f, maks) for f in files]
    return layanan.prediksi_massal(berkas, model_name=model)


@app.post(
    "/api/v1/predict-compare",
    tags=["prediksi"],
    response_model=ModelComparisonResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Berkas citra tidak valid"},
        503: {"model": ErrorResponse, "description": "Model belum termuat"},
    },
    summary="Komparasi prediksi ke 5 algoritma learner dari skema Orange (Test & Score)",
)
async def predict_compare(
    file: UploadFile = File(..., description="Citra sperma (.jpg, .jpeg, .png, .bmp)"),
) -> ModelComparisonResponse:
    if layanan is None:
        raise ModelBelumSiap("Server belum selesai memuat model.")
    nama, data = await _baca(file, settings.max_upload_bytes)
    return layanan.prediksi_komparasi(nama, data)


@app.post(
    "/predict",
    tags=["kompatibilitas"],
    response_model=LegacyPredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Berkas citra tidak valid"},
        503: {"model": ErrorResponse, "description": "Model belum termuat"},
    },
    summary="Endpoint kompatibilitas untuk sperm_classification_api.py (Gemini)",
)
async def legacy_predict(
    file: UploadFile = File(..., description="Citra sperma (.jpg, .jpeg, .png, .bmp)"),
    model: Optional[str] = Query(None, description="Pilihan model (opsional)"),
) -> LegacyPredictResponse:
    """Endpoint yang kompatibel 100% dengan format kembalian sperm_classification_api.py."""
    if layanan is None:
        raise ModelBelumSiap("Server belum selesai memuat model.")
    nama, data = await _baca(file, settings.max_upload_bytes)
    res = layanan.prediksi_satu(nama, data, model_name=model)
    return LegacyPredictResponse(
        status="success",
        prediction=res.predicted_class,
        confidence=f"{res.confidence * 100:.2f}%",
        embedding_size=1000,
        model_used=res.model_used,
    )