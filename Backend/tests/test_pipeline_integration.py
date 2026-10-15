"""Uji integrasi pipeline penuh dengan model SqueezeNet Orange (ndf) sungguhan.

Memverifikasi:
  1. embedder SqueezeNet Orange (ndf) siap dan menghasilkan fitur 1000-dim
  2. prapemrosesan Orange (227x227, RGB->BGR, mean Caffe)
  3. prediksi batch benar dan probabilitas berjumlah 1
  4. galat dimensi fitur terdeteksi

Jalankan:  python tests/test_pipeline_integration.py
Bukan bagian dari `pytest -q` (butuh paket ndf & bobot SqueezeNet).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings                       # noqa: E402
from app.services.pipeline import KlasifikasiService      # noqa: E402
from app.utils.errors import CitraTidakValid, InferensiGagal  # noqa: E402


def buat_png(ukuran=(400, 400)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", ukuran, (180, 165, 150)).save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    settings = get_settings()
    svc = KlasifikasiService(settings)

    print("1) Embedder SqueezeNet Orange (ndf)…")
    if not svc.extractor.siap:
        print(f"   GAGAL: {svc.extractor.pesan}")
        return 1
    print(f"   siap pada '{svc.device}', fitur {svc.extractor.dimensi}-dim")

    print("2) Status predictor")
    if svc.predictor.siap:
        print(f"   {svc.predictor.pesan}")
    else:
        print(f"   {svc.predictor.pesan}")

    print("3) Prapemrosesan citra (Orange)")
    from app.utils.image_utils import buka_dan_preprocess
    t = buka_dan_preprocess(buat_png(), "a.png", settings)
    print(f"   bentuk {t.shape}, dtype {t.dtype}, mean {t.mean():.4f}")
    assert t.shape == (227, 227, 3), t.shape
    assert t.dtype == np.float32

    print("4) Ekstraksi fitur SqueezeNet Orange")
    batch = [t, t, t]
    fitur = svc.extractor.ekstrak(batch)
    print(f"   bentuk {fitur.shape}")
    assert fitur.shape == (3, 1000), fitur.shape
    assert np.isfinite(fitur).all(), "fitur mengandung NaN/Inf"

    print("5) Siapkan model prediksi")
    if not svc.predictor.siap:
        # Placeholder dilatih pada fitur acak 1000-dim supaya langkah prediksi
        # tetap bisa diperiksa tanpa model terlatih.
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(8,), max_iter=5)
        m.fit(np.random.rand(6, 1000).astype(np.float32),
              ["Normal", "Abnormal", "Non-sperm"] * 2)
        svc.predictor._models["mlp"] = m
        svc.predictor._labels["mlp"] = ("Normal", "Abnormal", "Non-sperm")
        svc.predictor._primary_id = "mlp"
        svc.predictor._siap = True
        print("   MLP placeholder dilatih pada fitur acak 1000-dim")

    print("6) Prediksi batch")
    hasil = svc.predictor.prediksi(list(fitur))
    for i, item in enumerate(hasil):
        total = sum(item["probabilities"].values())
        print(f"   [{i}] {item['label']:>10}  conf={item['confidence']:.4f}  "
              f"sum={total:.6f}")
        assert item["label"] in ("Normal", "Abnormal", "Non-sperm")
        assert abs(total - 1.0) < 1e-3, total
    assert len(hasil) == 3

    print("7) Pipeline penuh (prediksi_massal)")
    respons = svc.prediksi_massal([("a.png", buat_png()), ("b.jpg", buat_png((200, 300)))])
    for r in respons:
        print(f"   {r.filename}: {r.predicted_class} "
              f"{r.confidence:.4f} {r.processing_time_ms:.1f}ms")
    assert len(respons) == 2

    print("8) Dimensi fitur salah terdeteksi")
    salah = [np.zeros((500,), dtype=np.float32)]
    try:
        svc.predictor.prediksi(salah)
        print("   GAGAL: dimensi salah tidak tertangkap")
        return 1
    except InferensiGagal as e:
        print(f"   tertangkap: {e}")

    print("9) Format terlarang ditolak")
    try:
        svc.prediksi_massal([("x.txt", buat_png())])
        print("   GAGAL: .txt tidak tertolak")
        return 1
    except CitraTidakValid as e:
        print(f"   tertangkap: {e}")

    print("10) Citra rusak ditolak")
    try:
        svc.prediksi_massal([("x.png", b"bukan gambar sama sekali")])
        print("   GAGAL: data rusak tidak tertolak")
        return 1
    except CitraTidakValid as e:
        print(f"   tertangkap: {e}")

    print("\nSEMUA PEMERIKSAAN INTEGRASI LULUS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
