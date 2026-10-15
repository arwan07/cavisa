/* ============================================================================
 * Konfigurasi CAVISA — Frontend
 * --------------------------------------------------------------------------
 * File ini dimuat SEBELUM app.js. Ubah nilai di sini, tidak perlu menyentuh
 * logika aplikasi.
 *
 * Kesesuaian dengan Backend (lihat Backend/Spec.md):
 *   GET  /health                → status server & model
 *   POST /api/v1/predict        → field "file" (multipart)
 *   POST /api/v1/predict-batch  → field "files" (multipart, list)
 *
 * Bentuk respons /api/v1/predict:
 *   {
 *     "filename": "sperma-01.jpg",
 *     "predicted_class": "Normal" | "Abnormal" | "Non-sperm",
 *     "confidence": 0.0 - 1.0,
 *     "probabilities": { "Normal": 0.0, "Abnormal": 0.0, "Non-sperm": 0.0 },
 *     "processing_time_ms": 123.4
 *   }
 * ========================================================================== */

window.CAVISA_CONFIG = {

  /* --- Koneksi backend ---------------------------------------------- */
  // Alamat URL backend FastAPI:
  // - Pengujian lokal (port 8001): otomatis ke "http://localhost:8000"
  // - Server produksi dengan Nginx Reverse Proxy: otomatis ke window.location.origin (port 80/443)
  // - Server terpisah: ubah manual ke "http://IP_SERVER:8000" atau "https://api.domainanda.com"
  apiBase: (typeof window !== "undefined" && (window.location.port === "8001" || window.location.hostname === "localhost"))
    ? "http://localhost:8000"
    : ((typeof window !== "undefined" && window.location.origin && window.location.origin !== "null") ? window.location.origin : "http://localhost:8000"),

  batasWaktuMs: 30000,   // abort bila server tidak merespons (30 detik)

  /* --- Batas unggah ----------------------------------------------- */
  maksUkuranMB: 10,
  maksBerkas: 12,
  maksBatch: 20,         // batas berkas untuk predict-batch

  /* --- Kunci penyimpanan lokal ------------------------------------- */
  StoreKey:      "cavisa.riwayat.v2",
  StoreKeyTheme: "cavisa.tema.v2",
  StoreKeyForm:  "cavisa.form.v2",
  StoreKeyServer:"cavisa.server.v2"
};
window.SIKSSP_CONFIG = window.CAVISA_CONFIG;