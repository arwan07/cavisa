# CAVISA (Computer-Aided Visual Sperm Analysis)

**CAVISA** adalah sistem cerdas berbasis kecerdasan buatan (*AI*) untuk klasifikasi morfologi sel sperma manusia ke dalam kategori **Normal**, **Abnormal**, dan **Non-Sperm**. 

Sistem ini mereproduksi secara identik pipeline riset **Orange Data Mining** (`SMIDS.ows`):
* **Feature Extractor:** *SqueezeNet v1.0 Orange (`ndf`)* — Lapisan pra-softmax `flatten10` menghasilkan vektor embedding **1000 dimensi**.
* **5 Algoritma Machine Learning (Orange Learners):**
  1. **Neural Network / Multi-Layer Perceptron (MLP)** — *Model Terbaik Paper (F1 0.854)*
  2. **Support Vector Machine (SVM)**
  3. **Random Forest (RF)**
  4. **k-Nearest Neighbors (k-NN)**
  5. **Naive Bayes (NB)**
* **Dukungan Format Citra:** Menerima format mikroskopis asli **BMP**, serta **JPG**, **JPEG**, dan **PNG** secara *native* tanpa kompresi *lossy*.

---

## Arsitektur Sistem

```text
┌────────────────────────────────────────────────────────┐
│                   KLIEN / BROWSER                      │
├───────────────────────────┬────────────────────────────┤
│   Portal Pengguna         │   Panel Administrator      │
│   (Frontend/index.html)   │   (Frontend/admin.html)    │
│   • Input data pasien     │   • Monitoring status server│
│   • Unggah citra mikroskop│   • Ganti 5 model aktif    │
│   • Hasil diagnosis badge │   • Manajemen dataset      │
│   • Riwayat & ekspor CSV  │   • Latih ulang & konfusi  │
└─────────────┬─────────────┴──────────────┬─────────────┘
              │                            │
              │  HTTP REST API             │
              ▼                            ▼
┌────────────────────────────────────────────────────────┐
│       NGINX REVERSE PROXY (Port 80 / 443 HTTPS)        │
│   • Menyajikan berkas statis frontend                  │
│   • Mem-proxy /api/ dan /health ke backend FastAPI     │
│   • Zero CORS issue & kompresi gzip                    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│              BACKEND FASTAPI (Port 8000)               │
│                                                        │
│  1. Prapemrosesan Caffe (Resize 227×227, BGR, Mean Sub)│
│  2. SqueezeNet Orange (ndf) -> Vektor Fitur 1000-dim   │
│  3. Model Klasifikasi (MLP / SVM / RF / kNN / NB)      │
│  4. Service Pelatihan Ulang (Trainer Service)          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   DATASET REPOSITORI                   │
│   ├── data_latih/     (80% - Pelatihan Model)          │
│   ├── data_validasi/  (10% - Validasi Parameter)       │
│   ├── data_uji/       (10% - Evaluasi & Uji Mandiri)   │
│   └── data_asli/      (Master Dataset SMIDS)           │
│       └── [Subfolder: Abnormal_Sperm, Non-Sperm,       │
│                       Normal_Sperm]                    │
└────────────────────────────────────────────────────────┘
```

---

## Panduan Cepat Menjalankan di Komputer Lokal (Development)

### 1. Jalankan Backend (FastAPI)
```bash
cd Backend

# Buat dan aktifkan virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate  # Di Windows: .venv\Scripts\activate

# Pasang dependensi
pip install -r requirements.txt

# Jalankan server FastAPI
uvicorn app.main:app --reload --port 8000
```
Backend siap di `http://localhost:8000` (Dokumentasi Swagger di `http://localhost:8000/docs`).

### 2. Jalankan Frontend
Buka terminal baru:
```bash
cd Frontend
python3 -m http.server 8001
```
Akses antarmuka melalui peramban:
* **Portal Pengguna (Klasifikasi):** `http://localhost:8001/`
* **Panel Admin (Pelatihan & Kontrol Model):** `http://localhost:8001/admin.html`

---

## Panduan Deployment ke Server Produksi (Production Deployment)

Tersedia dua opsi deployment ke server (VPS Linux seperti Ubuntu/Debian atau Cloud Server):

### OPSI A: Menggunakan Docker Compose (Paling Cepat & Mudah)

Proyek ini telah dilengkapi konfigurasi `docker-compose.yml` siap pakai yang mencakup kontainer backend FastAPI dan kontainer frontend Nginx Alpine.

1. **Pastikan Docker & Docker Compose sudah terpasang** di server Anda:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose-plugin
   ```

2. **Kloning repositori proyek**:
   ```bash
   git clone <URL_REPOSITORY_ANDA> /var/www/cavisa
   cd /var/www/cavisa
   ```

3. **Jalankan aplikasi dengan satu perintah**:
   ```bash
   docker compose up -d --build
   ```

4. **Selesai!** Aplikasi langsung aktif:
   * Portal Pengguna: `http://IP_SERVER/`
   * Portal Admin: `http://IP_SERVER/admin.html`
   * API Backend & Swagger: `http://IP_SERVER/docs`

Untuk menghentikan kontainer:
```bash
docker compose down
```

---

### OPSI B: Menggunakan Linux VPS Standalone (Systemd + Nginx + Let's Encrypt SSL)

Opsi ini memberikan performa terbaik, kendali penuh terhadap sistem, dan integrasi sertifikat HTTPS resmi.

#### Langkah 1: Persiapan Server Linux
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip nginx git curl certbot python3-certbot-nginx
```

#### Langkah 2: Salin Proyek & Setup Python Virtual Environment
```bash
sudo mkdir -p /var/www/cavisa
sudo chown -R $USER:$USER /var/www/cavisa
git clone <URL_REPOSITORY_ANDA> /var/www/cavisa

cd /var/www/cavisa/Backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

#### Langkah 3: Setup Backend dengan Systemd Service
Buat service unit agar backend berjalan otomatis dan restart saat reboot:
```bash
sudo nano /etc/systemd/system/cavisa-backend.service
```

Isi dengan konfigurasi berikut:
```ini
[Unit]
Description=CAVISA Sperm Classification Backend (FastAPI + Gunicorn)
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
    app.main:app

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Atur izin berkas dan jalankan servicenya:
```bash
sudo chown -R www-data:www-data /var/www/cavisa
sudo systemctl daemon-reload
sudo systemctl enable cavisa-backend
sudo systemctl start cavisa-backend
sudo systemctl status cavisa-backend
```

#### Langkah 4: Setup Nginx sebagai Web Server & Reverse Proxy
Buat konfigurasi situs Nginx:
```bash
sudo nano /etc/nginx/sites-available/cavisa
```

Isi dengan konfigurasi produksi:
```nginx
server {
    listen 80;
    server_name cavisa.domainanda.com;  # Ganti dengan domain atau IP server

    client_max_body_size 25M;
    root /var/www/cavisa/Frontend;
    index index.html;

    gzip on;
    gzip_types text/plain text/css application/json application/javascript image/svg+xml;

    # Portal Pengguna
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Portal Admin
    location /admin {
        try_files $uri $uri/ /admin.html;
    }

    # Aset Statis
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|bmp)$ {
        expires 7d;
        add_header Cache-Control "public, no-transform";
    }

    # Reverse Proxy ke Backend FastAPI (port 8000)
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    location /openapi.json {
        proxy_pass http://127.0.0.1:8000/openapi.json;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }
}
```

Aktifkan konfigurasi:
```bash
sudo ln -s /etc/nginx/sites-available/cavisa /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### Langkah 5: Pasang SSL HTTPS Gratis (Certbot)
```bash
sudo certbot --nginx -d cavisa.domainanda.com
```
Certbot akan otomatis mengonfigurasi sertifikat SSL gratis dari Let's Encrypt dan mengatur perpanjangan otomatis (*auto-renewal*).

---

## Ringkasan Endpoint API

| Metode | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/health` | Status server dan kesiapan model AI |
| `POST` | `/api/v1/predict` | Klasifikasi satu citra (mendukung format `.bmp`, `.jpg`, `.png`) |
| `POST` | `/api/v1/predict-batch` | Klasifikasi banyak citra sekaligus |
| `POST` | `/api/v1/predict-compare` | Komparasi hasil ke seluruh 5 algoritma Orange (Test & Score) |
| `GET` | `/api/v1/models` | Daftar 5 model algoritma Orange beserta status aktif |
| `POST` | `/api/v1/models/active` | Mengganti model aktif secara langsung tanpa reload server |
| `POST` | `/api/v1/models/reload` | Memuat ulang seluruh berkas model dari disk |
| `GET` | `/api/v1/datasets` | Daftar informasi dataset server (`data_latih`, `data_validasi`, dll.) |
| `POST` | `/api/v1/train` | Memulai proses pelatihan model dari folder tertentu |
| `GET` | `/api/v1/train/status` | Mengambil status progres dan metrik evaluasi pelatihan |
| `POST` | `/predict` | Endpoint kompatibilitas penuh (Legacy API format) |

---

## Struktur Dataset & Pelatihan

Struktur repositori dataset disusun rapi sesuai standar data science (proporsi 80:10:10):
```text
├── data_asli/      # Master dataset SMIDS
├── data_latih/     # 80% sampel untuk melatih model
├── data_validasi/  # 10% sampel untuk validasi
└── data_uji/       # 10% sampel untuk pengujian mandiri
    ├── Abnormal_Sperm/
    ├── Non-Sperm/
    └── Normal_Sperm/
```
Anda dapat melatih ulang kelima model kapan saja melalui **Panel Admin** di browser atau melalui CLI:
```bash
cd Backend
python train_models.py --data-dir "../data_latih"
```

---

## Lisensi & Batasan

Aplikasi ini ditujukan sebagai alat bantu analisis riset laboratorium dan edukasi. Hasil klasifikasi otomatis disajikan untuk membantu penapisan awal dan tetap memerlukan verifikasi tenaga ahli laboratorium medis untuk penegakan diagnosis klinis resmi.