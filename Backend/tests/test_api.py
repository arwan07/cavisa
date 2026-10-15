"""Pengujian lapisan HTTP memakai stub, tanpa torch atau model terlatih.

`KlasifikasiService` diganti objek dummy yang mengembalikan respons tetap,
sehingga endpoint, handler galat, dan validasi berkas bisa diuji cepat.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.schemas.prediction import DatasetFolderInfo, PredictionResponse, TrainStatusResponse
from app.services.pipeline import KlasifikasiService
from app.utils.errors import CitraTidakValid


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 110, 100)).save(buf, format="PNG")
    return buf.getvalue()


class StubTrainer:
    def list_datasets(self):
        return [
            DatasetFolderInfo(
                id="data_latih",
                name="Data Latih",
                path="/dummy/data_latih",
                exists=True,
                total_images=2400,
                counts={"normal_sperm": 817, "abnormal_sperm": 804, "non_sperm": 779},
            )
        ]

    def get_status(self):
        return TrainStatusResponse(
            is_training=False,
            progress_percent=100,
            current_stage="idle",
        )

    def start_training(self, req, on_success_reload_fn=None):
        return TrainStatusResponse(
            is_training=True,
            progress_percent=0,
            current_stage="Mulai pelatihan...",
            dataset_used=req.dataset_folder,
        )


class StubService(KlasifikasiService):
    """Melewati pemuatan model sungguhan."""

    def __init__(self, kelas="Normal", model_loaded=True):
        self._kelas = kelas
        self._loaded = model_loaded
        self.panggilan: list = []
        self.trainer = StubTrainer()

    # properti yang dipakai endpoint
    @property
    def model_loaded(self):
        return self._loaded

    @property
    def active_model(self):
        return "mlp"

    @property
    def kelas(self):
        return ["Normal", "Abnormal", "Non-sperm"]

    @property
    def device(self):
        return "cpu"

    @property
    def pesan_status(self):
        return "stub"

    def daftar_model(self):
        return [
            {
                "id": "mlp",
                "name": "Neural Network (MLP)",
                "description": "desc",
                "paper_table": "Table 5",
                "is_active": True,
                "is_loaded": True,
            },
            {
                "id": "svm",
                "name": "Support Vector Machine",
                "description": "desc",
                "paper_table": "Table 3",
                "is_active": False,
                "is_loaded": True,
            },
        ]

    def prediksi_satu(self, nama, data, model_name=None):
        self.panggilan.append((nama, data, model_name))
        return PredictionResponse(
            filename=nama,
            predicted_class=self._kelas,
            confidence=0.9123,
            probabilities={"Normal": 0.9123, "Abnormal": 0.0701, "Non-sperm": 0.0176},
            processing_time_ms=12.5,
            model_used=model_name or "mlp",
        )

    def prediksi_massal(self, berkas, model_name=None):
        return [self.prediksi_satu(n, d, model_name=model_name) for n, d in berkas]

    def prediksi_komparasi(self, nama, data):
        from app.schemas.prediction import ModelComparisonItem, ModelComparisonResponse

        return ModelComparisonResponse(
            filename=nama,
            processing_time_ms=15.0,
            best_prediction=self._kelas,
            comparisons=[
                ModelComparisonItem(
                    model_id="mlp",
                    model_name="Neural Network (MLP)",
                    predicted_class=self._kelas,
                    confidence=0.9123,
                    probabilities={"Normal": 0.9123, "Abnormal": 0.0701, "Non-sperm": 0.0176},
                )
            ],
        )

    def ringkasan(self):
        return {
            "model_loaded": self._loaded,
            "device": "cpu",
            "classes": self.kelas,
            "active_model": self.active_model,
            "available_models": ["mlp", "svm"],
            "message": "stub",
        }


@pytest.fixture
def stub(monkeypatch):
    s = StubService()
    monkeypatch.setattr("app.main.layanan", s)
    return s


@pytest.fixture
def client():
    """TestClient tanpa `with` → lifespan tidak dijalankan, jadi torch tidak dimuat."""
    from app.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health_model_siap(client, stub):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"


def test_health_model_belum_siap(client, monkeypatch):
    monkeypatch.setattr("app.main.layanan", StubService(model_loaded=False))
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


# ---------------------------------------------------------------------------
# /api/v1/predict
# ---------------------------------------------------------------------------
def test_predict_satu(client, stub):
    r = client.post("/api/v1/predict", files={"file": ("sperma.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "sperma.png"
    assert body["predicted_class"] == "Normal"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["probabilities"]["Normal"] == pytest.approx(0.9123)
    assert body["processing_time_ms"] >= 0
    # field konteks front-end tidak boleh menggagalkan request
    assert stub.panggilan and stub.panggilan[0][0] == "sperma.png"


def test_predict_dengan_field_konteks(client, stub):
    """Frontend mengirim nama_pengguna/tanggal; backend harus mengabaikannya."""
    r = client.post(
        "/api/v1/predict",
        files={"file": ("s.png", png_bytes(), "image/png")},
        data={"nama_pengguna": "Renita", "tanggal_pemeriksaan": "2026-10-06", "kode_sampel": "SP-1"},
    )
    assert r.status_code == 200


def test_predict_tanpa_field_file(client, stub):
    r = client.post("/api/v1/predict", data={"nama": "x"})
    assert r.status_code == 422  # validasi Pydantic FastAPI


def test_predict_berkas_kosong(client, stub):
    r = client.post("/api/v1/predict", files={"file": ("a.png", b"", "image/png")})
    assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# /api/v1/predict-batch
# ---------------------------------------------------------------------------
def test_predict_batch(client, stub):
    files = [
        ("files", ("a.png", png_bytes(), "image/png")),
        ("files", ("b.png", png_bytes(), "image/png")),
    ]
    r = client.post("/api/v1/predict-batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) == 2
    assert body[0]["filename"] == "a.png"
    assert body[1]["filename"] == "b.png"


def test_predict_batch_kosong(client, stub):
    r = client.post("/api/v1/predict-batch", files=[])
    assert r.status_code in (400, 422)


def test_predict_batch_melebihi_batas(client, stub, monkeypatch):
    from app.config import get_settings
    s = get_settings()
    monkeypatch.setattr(
        "app.main.settings",
        type(s)(**{**s.__dict__, "max_batch_files": 1}),
    )
    files = [
        ("files", ("a.png", png_bytes(), "image/png")),
        ("files", ("b.png", png_bytes(), "image/png")),
    ]
    r = client.post("/api/v1/predict-batch", files=files)
    assert r.status_code == 400
    assert "melebihi batas" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Galat domain
# ---------------------------------------------------------------------------
def test_galat_domain_jadi_json(client, monkeypatch):
    class Gagal(StubService):
        def prediksi_satu(self, nama, data, *args, **kwargs):
            raise CitraTidakValid("Berkas rusak.")

    monkeypatch.setattr("app.main.layanan", Gagal())
    r = client.post("/api/v1/predict", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 400
    assert r.json()["detail"] == "Berkas rusak."
    assert r.json()["code"] == "invalid_image"


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------
def test_root(client, stub):
    body = client.get("/").json()
    assert body["service"]
    assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# Skema Orange & Legacy Predict
# ---------------------------------------------------------------------------
def test_models_list(client, stub):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["active_model"] == "mlp"
    assert len(data["models"]) >= 2
    ids = [m["id"] for m in data["models"]]
    assert "mlp" in ids and "svm" in ids


def test_predict_with_model_param(client, stub):
    r = client.post("/api/v1/predict?model=svm", files={"file": ("sperma.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "svm"


def test_predict_compare(client, stub):
    r = client.post("/api/v1/predict-compare", files={"file": ("sperma.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "sperma.png"
    assert "comparisons" in body
    assert len(body["comparisons"]) >= 1
    assert body["comparisons"][0]["model_id"] == "mlp"


def test_legacy_predict(client, stub):
    r = client.post("/predict", files={"file": ("sperma.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["prediction"] == "Normal"
    assert "%" in body["confidence"]
    assert body["embedding_size"] == 1000


def test_list_datasets_endpoint(client, stub):
    r = client.get("/api/v1/datasets")
    assert r.status_code == 200
    body = r.json()
    assert "datasets" in body
    assert len(body["datasets"]) >= 1
    assert body["datasets"][0]["id"] == "data_latih"
    assert body["datasets"][0]["total_images"] == 2400


def test_train_endpoints(client, stub):
    r_status = client.get("/api/v1/train/status")
    assert r_status.status_code == 200
    assert r_status.json()["is_training"] is False

    r_start = client.post("/api/v1/train", json={"dataset_folder": "data_latih"})
    assert r_start.status_code == 200
    assert r_start.json()["is_training"] is True
    assert r_start.json()["dataset_used"] == "data_latih"