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

  apiBase: (function() {
    if (typeof window === "undefined" || !window.location.origin || window.location.origin === "null") {
      return "http://localhost:8000";
    }
    // Jika frontend dibuka dari web server terpisah untuk dev lokal (misal port 8001 / 5500 / 3000)
    if (window.location.hostname === "localhost" && (window.location.port === "8001" || window.location.port === "5500" || window.location.port === "3000")) {
      return "http://localhost:8000";
    }
    // Disajikan langsung oleh Backend (Railway, cloud host, atau localhost port sama)
    return window.location.origin;
  })(),

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