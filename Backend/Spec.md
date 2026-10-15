
```markdown
# Spesifikasi Backend & Master Prompt Vibe Coding
**API Klasifikasi Morfologi Sperma (SqueezeNet 1.1 + MLP Classifier)**

---

## 1. Arsitektur & Spesifikasi Sistem

* **Framework API:** FastAPI (Python 3.10+) dengan Uvicorn
* **Feature Extractor:** PyTorch `SqueezeNet 1.1` (Pretrained ImageNet)
  * **Input:** Citra RGB single-cell \\(224 \times 224\\) piksel
  * **Output:** Vektor fitur 1.000 dimensi (`1000-dim feature vector`)
* **Classifier:** Scikit-Learn `MLPClassifier`
  * **Input:** Vektor fitur 1.000 dimensi
  * **Output:** Label kelas (`Normal`, `Abnormal`, `Non-sperm`) & skor probabilitas
* **Validasi Data:** Pydantic v2
* **Pengolahan Citra:** Pillow (PIL) & Torchvision Transforms

---

## 2. Struktur Proyek (`sperm-classifier-backend`)

```text
sperm-classifier-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Entrypoint FastAPI & middleware CORS
│   ├── config.py               # Konfigurasi path model & parameter
│   ├── models/                 # Directory file model terlatih
│   │   └── mlp_model.pkl       # Model MLP terlatih (joblib)
│   ├── schemas/
│   │   └── prediction.py       # Schema Pydantic input/output
│   ├── services/
│   │   ├── feature_extractor.py # SqueezeNet Image Embedding (1000-dim)
│   │   └── predictor.py         # Inference & Scoring MLP
│   └── utils/
│       └── image_utils.py       # Preprocessing citra (Resize & Normalisasi)
├── requirements.txt            # Dependencies Python
├── Dockerfile                  # Containerization
└── README.md                   # Dokumentasi API
```

---

## 3. Master Prompt Vibe Coding

> **Instruksi:** Salin seluruh blok kode di bawah ini dan tempelkan ke AI Code Editor pilihan Anda (**Cursor**, **Windsurf**, **Claude Code**, **GitHub Copilot**, atau **Bolt.new**).

```text
You are an expert Senior Python Backend Engineer and Machine Learning Engineer.
Build a production-ready, clean, modular FastAPI backend for a Sperm Abnormality Classification API using SqueezeNet feature extraction and an MLP Classifier model.

### PROJECT REQUIREMENTS & TECH STACK:
- Python 3.10+
- Framework: FastAPI with Uvicorn
- Deep Learning & ML: PyTorch, Torchvision, Scikit-learn, Joblib, Pillow, Numpy
- Data Validation: Pydantic v2

### ML PIPELINE ARCHITECTURE:
1. Input: Single-cell sperm RGB image file upload (Formats: .jpg, .jpeg, .png).
2. Image Preprocessing (`app/utils/image_utils.py`):
   - Convert image bytes to PIL RGB.
   - Resize image to (224, 224).
   - Convert to PyTorch Tensor and normalize using ImageNet stats:
     mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225].
3. Feature Extraction (`app/services/feature_extractor.py`):
   - Load PyTorch SqueezeNet 1.1 (`models.squeezenet1_1(weights=DEFAULT)`).
   - Set model to `.eval()` mode and freeze parameters (`requires_grad = False`).
   - Extract a 1000-dimensional feature vector per image.
4. MLP Classification (`app/services/predictor.py`):
   - Load a pre-trained scikit-learn MLP model (`mlp_model.pkl`) using `joblib`.
   - Pass the 1000-dim feature vector into `mlp.predict_proba()` and `mlp.predict()`.
   - Map predictions to class labels: ['Normal', 'Abnormal', 'Non-sperm'].

### API ENDPOINTS TO IMPLEMENT:
1. `GET /health`
   - Returns JSON status of server and model preloading state.
2. `POST /api/v1/predict`
   - Accepts multipart/form-data image file (`file: UploadFile`).
   - Performs full pipeline: Preprocess -> Extract Features (SqueezeNet) -> Classify (MLP).
   - Returns JSON response:
     {
       "filename": str,
       "predicted_class": str,
       "confidence": float,
       "probabilities": {
         "Normal": float,
         "Abnormal": float,
         "Non-sperm": float
       },
       "processing_time_ms": float
     }
3. `POST /api/v1/predict-batch`
   - Accepts list of image files (`files: List[UploadFile]`).
   - Processes batch efficiently and returns array of prediction JSON objects.

### ADDITIONAL REQUIREMENTS:
- Handle exceptions gracefully (invalid image formats, corrupt files, missing model file). Return HTTP 400 or 500 with clear JSON detail messages.
- Add CORS middleware allowing all origins for frontend/mobile integration.
- Write a clean `requirements.txt` with appropriate package versions.
- Provide a ready-to-run `Dockerfile` using `python:3.10-slim`.
- Implement startup event (`@asynccontextmanager` / `lifespan`) to preload SqueezeNet and MLP model into memory once when the server starts.

Please generate the full project code structure file by file.
```

---

## 4. Langkah Penggunaan & Eksekusi

1. **Jalankan Vibe Coding:** Masukkan prompt di atas ke AI Code Editor.
2. **Letakkan Model:** Simpan file model hasil pelatihan Colab (`mlp_model.pkl`) ke dalam folder `app/models/`.
3. **Instalasi Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Jalankan Server Backend:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
5. **Uji API:** Buka browser dan akses `http://localhost:8000/docs` untuk mencoba upload foto sperma via Swagger UI.
```

---

