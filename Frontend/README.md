# Frontend — CAVISA (Computer-Aided Visual Sperm Analysis)

Antarmuka web modern, *mobile-first*, dan responsif untuk klasifikasi morfologi sel sperma manusia. 

Dibuat dengan **HTML5, CSS3, dan Vanilla JavaScript murni** — tanpa dependensi eksternal, tanpa node_modules, tanpa webpack/vite, sehingga sangat ringan, cepat, dan mudah di-deploy di web server mana pun.

---

## 2 Antarmuka Aplikasi

Aplikasi menyediakan dua antarmuka yang terpisah secara tegas sesuai peran:

```text
Frontend/
├── index.html            # [USER] Portal Klasifikasi Pemeriksaan Sel Sperma
├── admin.html            # [ADMIN] Panel Monitoring Server, Pelatihan Ulang & Dataset
├── admin/
│   └── index.html        # Redirect otomatis ke ../admin.html
├── nginx.conf            # Konfigurasi siap pakai untuk Nginx Server Produksi
└── assets/
    ├── css/style.css     # Sistem desain terpadu (tema terang/gelap, responsif)
    └── js/
        ├── config.js     # Konfigurasi alamat server backend (apiBase)
        ├── app.js        # Logika antarmuka pengguna & komunikasi API
        └── admin.js      # Logika kontrol server, pelatihan model & visualisasi metrik
```

### 1. Portal Pengguna (`index.html`)
* Dirancang khusus untuk staf laboratorium / operator medis.
* Formulir data pemeriksaan (Nama pengguna, Tanggal pemeriksaan, Kode sampel).
* Unggah citra mikroskopis tunggal maupun banyak sekaligus (format **BMP**, **JPG**, **PNG**).
* Hasil instan: Badge diagnosis (**Normal**, **Abnormal**, **Non-Sperm**), persentase keyakinan model, rincian kriteria, dan waktu inferensi server.
* Fitur riwayat pemeriksaan lokal (`localStorage`), filter per kelas, ekspor rekapitulasi data ke format **CSV**, dan cetak hasil pemeriksaan (*print-ready*).

### 2. Panel Administrator (`admin.html` atau rute `/admin`)
* Dirancang khusus untuk administrator sistem / peneliti.
* **Status Server & Model:** Menampilkan kondisi kesehatan server, model aktif saat ini, dan daftar 5 learner skema Orange (MLP, SVM, Random Forest, k-NN, Naive Bayes).
* **Penggantian Model Aktif:** Mengganti algoritma klasifikasi secara live dengan 1 klik tanpa perlu me-restart server.
* **Manajemen Dataset Server:** Memindai dan menampilkan jumlah sampel citra pada folder dataset (`data_latih`, `data_validasi`, `data_uji`, `data_asli`).
* **Pelatihan Ulang Terintegrasi:** Menjalankan pelatihan model langsung dari browser dengan parameter split data dan batas sampel.
* **Visualisasi Metrik:** Menampilkan progress ekstraksi fitur SqueezeNet secara real-time, ringkasan skor akurasi/F1, dan matriks konfusi (*confusion matrix*) lengkap untuk tiap model.

---

## Panduan Deployment ke Server Produksi

### Metode 1: Menggunakan Nginx (Direkomendasikan)
Metode ini adalah standar industri terbaik. Nginx bertindak sebagai web server berkas statis frontend sekaligus *Reverse Proxy* untuk backend FastAPI. Dengan cara ini, Anda tidak akan pernah mengalami masalah **CORS** karena frontend dan backend berada dalam satu origin/domain yang sama.

#### 1. Salin Berkas Frontend ke Server
Salin seluruh folder `Frontend/` ke direktori web server (misalnya `/var/www/cavisa/Frontend`).

#### 2. Konfigurasi Nginx
Gunakan template konfigurasi yang telah disediakan di `Frontend/nginx.conf` atau buat file baru:
```bash
sudo nano /etc/nginx/sites-available/cavisa
```

Isi dengan konfigurasi berikut (sesuaikan `server_name` dan path `root`):
```nginx
server {
    listen 80;
    server_name cavisa.domainanda.com;  # atau IP server Anda

    # Batas ukuran upload file citra
    client_max_body_size 25M;

    # Direktori berkas Frontend
    root /var/www/cavisa/Frontend;
    index index.html;

    # Kompresi gzip
    gzip on;
    gzip_types text/plain text/css application/json application/javascript image/svg+xml;

    # Halaman User
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Halaman Admin
    location /admin {
        try_files $uri $uri/ /admin.html;
    }

    # Cache aset statis
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg|bmp)$ {
        expires 7d;
        add_header Cache-Control "public, no-transform";
    }

    # Proxy ke Backend FastAPI (port 8000)
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

Aktifkan konfigurasi dan restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/cavisa /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 3. Pasang SSL/HTTPS Gratis (Let's Encrypt)
Jika Anda menggunakan domain, aktifkan HTTPS dengan Certbot:
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d cavisa.domainanda.com
```

---

### Metode 2: Menggunakan Docker / Docker Compose
Jika menggunakan Docker, frontend otomatis dilayani oleh kontainer Nginx Alpine resmi.

Jalankan dari direktori root proyek:
```bash
docker compose up -d
```
Frontend langsung dapat diakses pada port 80:
* Pengguna: `http://IP_SERVER/`
* Administrator: `http://IP_SERVER/admin.html`

---

### Metode 3: Hosting Statis Terpisah (Vercel, Netlify, Apache, S3)
Bila Frontend diletakkan pada penyedia hosting statis terpisah dari server backend:

1. Buka berkas `assets/js/config.js`.
2. Ubah `apiBase` mengarah ke alamat URL publik backend Anda:
   ```javascript
   window.CAVISA_CONFIG = {
     apiBase: "https://api-cavisa.domainanda.com", // atau "http://IP_SERVER:8000"
     ...
   };
   ```
3. Pastikan pada backend (`Backend/.env`), `CORS_ORIGINS` sudah mencantumkan domain frontend Anda agar request tidak diblokir browser.

---

## Konfigurasi Frontend (`assets/js/config.js`)

Seluruh pengaturan koneksi dan batasan unggahan terpusat pada berkas `assets/js/config.js`:

| Parameter | Deskripsi | Nilai Default |
|---|---|---|
| `apiBase` | Alamat URL backend FastAPI | Otomatis menyesuaikan (localhost / origin) |
| `batasWaktuMs` | Batas waktu tunggu request sebelum timeout | `30000` (30 detik) |
| `maksUkuranMB` | Batas ukuran per berkas citra | `10` MB |
| `maksBerkas` | Batas maksimum citra dalam antrean per sesi | `12` berkas |
| `maksBatch` | Batas maksimum berkas untuk analisis batch | `20` berkas |

---

## Pengujian Lokal

Untuk menguji frontend di komputer lokal Anda:
```bash
cd Frontend
python3 -m http.server 8001
```
* Buka antarmuka pengguna: `http://localhost:8001`
* Buka antarmuka admin: `http://localhost:8001/admin.html`