# Backend — API Klasifikasi Morfologi Sel Sperma (FastAPI + SqueezeNet Orange)

Backend API untuk klasifikasi morfologi sel sperma manusia berbasis pipeline **Orange Data Mining** (`SMIDS.ows`).

Pipeline inferensi:
```text
Citra Masukan (BMP / JPG / PNG)
      │
      ▼
Pillow (Resize 227×227 LANCZOS, RGB→BGR, Caffe Mean Subtraction)
      │
      ▼
SqueezeNet Orange/ndf (Ekstraksi Fitur Lapisan pra-softmax flatten10 → 1000 Dimensi)
      │
      ▼
5 Algoritma Machine Learning (MLP, SVM, Random Forest, k-NN, Naive Bayes)
      │
      ▼
Prediksi Kelas: Normal / Abnormal / Non-sperm (+ Probabilitas Tiap Kelas)
```

---

## 5 Algoritma Learner (Sesuai Skema Orange & Paper)

| ID Model | Nama Algoritma | Parameter Utama | Catatan Riset |
|---|---|---|---|
| `mlp` | **Neural Network (MLP)** | Hidden Layer: 128, Activation: ReLU, Solver: Adam, Max Iter: 200 | **Model Terbaik (F1 0.854)** *(Default)* |
| `svm` | **Support Vector Machine (SVM)** | Cost: 1.0, Kernel: RBF, Max Iter: 100, Probability: True | Klasifikasi margin tinggi |
| `rf` | **Random Forest (RF)** | Trees: 100, Min Subset Split: 5 | Ensemble bagging |
| `knn` | **k-Nearest Neighbors (kNN)** | k: 5, Metric: Euclidean, Weights: Uniform | Non-parametrik |
| `nb` | **Naive Bayes (NB)** | GaussianNB | Probabilistik cepat |

---

## Struktur Direktori

```text
Backend/
├── app/
│   ├── main.py                   # FastAPI app, CORS, lifespan, routing endpoint
│   ├── config.py                 # Konfigurasi aplikasi & environment (.env)
│   ├── models/                   # Direktori penyimpanan berkas model terlatih (.pkl)
│   │   ├── mlp_model.pkl         # Model utama (Multi-Layer Perceptron)
│   │   ├── svm_model.pkl         # Model SVM
│   │   ├── rf_model.pkl          # Model Random Forest
│   │   ├── knn_model.pkl         # Model k-NN
│   │   └── nb_model.pkl          # Model Naive Bayes
│   ├── schemas/
│   │   └── prediction.py         # Skema Pydantic v2 (Request, Response, Train)
│   ├── services/
│   │   ├── feature_extractor.py  # SqueezeNet Orange (ndf) 1000-dim
│   │   ├── models_factory.py     # Builder & spesifikasi 5 algoritma Orange
│   │   ├── predictor.py          # Multi-model inference engine (scikit-learn)
│   │   ├── pipeline.py           # Orkestrator pra-proses → fitur → inferensi
│   │   └── trainer.py            # Service pelatihan ulang & evaluasi dataset
│   └── utils/
│       ├── image_utils.py        # Prapemrosesan Caffe SqueezeNet (identik Orange)
│       └── errors.py             # Error handler domain terpusat
├── Dockerfile                    # Container image backend
├── requirements.txt              # Dependensi produksi (termasuk gunicorn)
├── requirements-dev.txt          # Dependensi testing (pytest, flake8)
├── .env.example                  # Template variabel lingkungan
└── README.md                     # Berkas dokumentasi ini
```

---

## Panduan Deployment ke Server Produksi (Linux / Ubuntu / Debian)

### 1. Kebutuhan Sistem
* **OS:** Ubuntu 22.04 LTS / 24.04 LTS atau Debian 12 (disarankan)
* **Python:** Python 3.10, 3.11, atau 3.12
* **RAM:** Minimal 2 GB (4 GB disarankan untuk proses pelatihan ulang di background)
* **CPU:** 2 Core atau lebih (SqueezeNet Orange berjalan murni di CPU dengan pustaka NumPy)
* **Penyimpanan:** Minimal 10 GB ruang kosong untuk dataset & model

### 2. Instalasi Dependensi Sistem
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip curl git
```

### 3. Kloning Proyek & Setup Virtual Environment
```bash
# Pindah ke direktori deployment
cd /var/www  # atau direktori pilihan Anda
git clone <URL_REPOSITORY_ANDA> cavisa
cd cavisa/Backend

# Buat virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Pasang dependensi
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Konfigurasi Variabel Lingkungan (`.env`)
Salin file konfigurasi contoh:
```bash
cp .env.example .env
nano .env
```
Sesuaikan parameter penting untuk produksi:
```ini
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
# Ganti dengan domain frontend Anda (atau '*' bila memakai reverse proxy)
CORS_ORIGINS=http://localhost:8001,https://cavisa.domainanda.com
ACTIVE_MODEL=mlp
AUTO_TRAIN_BASELINE=true
MAX_UPLOAD_MB=15
MAX_BATCH_FILES=30
```

### 5. Setup Service Otomatis dengan Systemd
Agar server backend berjalan otomatis saat server menyala (*boot*) dan me-restart sendiri jika terjadi galat, buat service unit Systemd:

```bash
sudo nano /etc/systemd/system/cavisa-backend.service
```

Isi berkas tersebut dengan konfigurasi berikut (sesuaikan path `/var/www/cavisa`):
```ini
[Unit]
Description=CAVISA Sperm Classification FastAPI Backend
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/cavisa/Backend
Environment="PATH=/var/www/cavisa/Backend/.venv/bin"
EnvironmentFile=/var/www/cavisa/Backend/.env
ExecStart=/var/www/cavisa/Backend/.venv/bin/gunicorn \
    -w 2 \
    -k uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 \
    --timeout 120 \
    --access-logfile /var/log/cavisa-access.log \
    --error-logfile /var/log/cavisa-error.log \
    app.main:app

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Pastikan izin folder tepat, lalu aktifkan servicenya:
```bash
# Berikan izin ke www-data
sudo chown -R www-data:www-data /var/www/cavisa

# Reload systemd dan jalankan service
sudo systemctl daemon-reload
sudo systemctl enable cavisa-backend
sudo systemctl start cavisa-backend

# Cek status service
sudo systemctl status cavisa-backend
```

Untuk memantau log aplikasi secara real-time:
```bash
sudo journalctl -u cavisa-backend -f
```

---

## Deployment Menggunakan Docker

Bila server Anda sudah terpasang Docker:

```bash
cd Backend

# 1. Build image Docker
docker build -t cavisa-backend:latest .

# 2. Jalankan container dengan volume model
docker run -d \
  --name cavisa-backend \
  --restart unless-stopped \
  -p 8000:8000 \
  -v "$(pwd)/app/models:/srv/app/models" \
  cavisa-backend:latest
```

Cek status container:
```bash
docker ps
curl http://localhost:8000/health
```

---

## Dokumentasi Endpoint API

Dokumentasi interaktif OpenAPI (Swagger UI) dapat diakses langsung pada:
`http://ALAMAT_SERVER:8000/docs`

### 1. Kesehatan Server (`GET /health`)
```bash
curl -s http://localhost:8000/health
```
```json
{
  "status": "ok",
  "service": "Sperm Abnormality Classification API",
  "version": "1.0.0",
  "model_loaded": true,
  "device": "cpu",
  "classes": ["Abnormal", "Non-sperm", "Normal"],
  "active_model": "mlp",
  "available_models": ["mlp", "svm", "rf", "knn", "nb"],
  "message": "Model siap melayani prediksi (Aktif: MLP)."
}
```

### 2. Klasifikasi Citra Tunggal (`POST /api/v1/predict`)
Menerima citra format `.bmp`, `.jpg`, `.jpeg`, `.png`:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -F "file=@contoh_sperma.bmp"
```
Opsional: tentukan model secara spesifik via query parameter (`?model=svm`):
```bash
curl -X POST "http://localhost:8000/api/v1/predict?model=svm" \
     -F "file=@contoh_sperma.bmp"
```
Respons:
```json
{
  "filename": "contoh_sperma.bmp",
  "predicted_class": "Normal",
  "confidence": 0.943391,
  "probabilities": {
    "Abnormal": 0.054447,
    "Non-sperm": 0.002162,
    "Normal": 0.943391
  },
  "processing_time_ms": 45.2,
  "model_used": "mlp"
}
```

### 3. Klasifikasi Massal (`POST /api/v1/predict-batch`)
```bash
curl -X POST "http://localhost:8000/api/v1/predict-batch" \
     -F "files=@sel1.bmp" \
     -F "files=@sel2.bmp"
```

### 4. Komparasi 5 Model (`POST /api/v1/predict-compare`)
Menjalankan inferensi ke seluruh 5 algoritma learner Orange sekaligus (Test & Score):
```bash
curl -X POST "http://localhost:8000/api/v1/predict-compare" \
     -F "file=@contoh_sperma.bmp"
```

### 5. Manajemen Model (`GET /api/v1/models` & `POST /api/v1/models/active`)
* **Daftar model:** `GET /api/v1/models`
* **Ganti model aktif tanpa restart server:**
  ```bash
  curl -X POST "http://localhost:8000/api/v1/models/active?model=rf"
  ```
* **Reload file model dari disk:** `POST /api/v1/models/reload`

### 6. Manajemen Dataset & Pelatihan (`/api/v1/datasets`, `/api/v1/train`)
* **Daftar dataset:** `GET /api/v1/datasets`
* **Mulai pelatihan dari folder:**
  ```bash
  curl -X POST "http://localhost:8000/api/v1/train" \
       -H "Content-Type: application/json" \
       -d '{"folder": "data_latih", "test_split": 0.2, "random_state": 42}'
  ```
* **Cek status progres pelatihan:** `GET /api/v1/train/status`

---

## Pengujian Otomatis (Testing)

Jalankan test suite untuk memastikan seluruh komponen berfungsi:
```bash
pytest -q
```
Hasil verifikasi pengujian: seluruh 26 tes unit dan pengujian integrasi prapemrosesan BGR serta model Orange berstatus lulus (*pass*).