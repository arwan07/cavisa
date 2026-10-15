"""Pengujian yang tidak memerlukan model terlatih.

Cakupan: prapemrosesan citra (SqueezeNet Orange) dan pemetaan label.
Lapisan HTTP diuji di `test_api.py` memakai stub.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.config import CLASS_LABELS, canonical_label, get_settings
from app.utils.errors import CitraTidakValid, FileKosong
from app.utils.image_utils import buka_dan_preprocess

settings = get_settings()


def buat_png(ukuran=(320, 320), warna=(200, 190, 180)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", ukuran, warna).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# image_utils
# ---------------------------------------------------------------------------
def test_png_preprocess_bentuk_dan_nilai():
    arr = buka_dan_preprocess(buat_png(), "a.png", settings)
    assert arr.shape == (227, 227, 3)          # HWC, BGR
    assert arr.dtype == np.float32
    # Di Orange, piksel (R,G,B)=(200,190,180) dikurangi mean Caffe [104.006, 116.669, 122.679].
    c0, c1, c2 = arr[0, 0]
    assert c0 == pytest.approx(200 - 104.006, abs=1e-2)
    assert c1 == pytest.approx(190 - 116.669, abs=1e-2)
    assert c2 == pytest.approx(180 - 122.679, abs=1e-2)


def test_jpg_diterima():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), (10, 20, 30)).save(buf, format="JPEG")
    arr = buka_dan_preprocess(buf.getvalue(), "a.jpg", settings)
    assert arr.shape == (227, 227, 3)


def test_ekstensi_terlarang_ditolak():
    with pytest.raises(CitraTidakValid):
        buka_dan_preprocess(buat_png(), "a.txt", settings)


def test_berkas_bukan_gambar_ditolak():
    with pytest.raises(CitraTidakValid):
        buka_dan_preprocess(b"ini bukan gambar", "a.png", settings)


def test_berkas_kosong_ditolak():
    with pytest.raises(FileKosong):
        buka_dan_preprocess(b"", "a.png", settings)


def test_transparan_ditempel_ke_latar_putih():
    buf = io.BytesIO()
    Image.new("RGBA", (50, 50), (255, 0, 0, 0)).save(buf, format="PNG")
    arr = buka_dan_preprocess(buf.getvalue(), "a.png", settings)
    # Semua piksel tembus pandang -> ditempel ke latar putih (255) -> nilai BGR
    # setelah dikurangi mean Caffe bernilai positif.
    assert arr.mean() > 0
    b, g, r = arr[0, 0]
    assert b == pytest.approx(255 - 104.006, abs=1e-2)
    assert g == pytest.approx(255 - 116.669, abs=1e-2)
    assert r == pytest.approx(255 - 122.679, abs=1e-2)


def test_dimensi_rasio_berbeda_tetap_227():
    arr = buka_dan_preprocess(buat_png((800, 200)), "panj.png", settings)
    assert arr.shape == (227, 227, 3)


# ---------------------------------------------------------------------------
# config: pemetaan label
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("masukan,harapan", [
    ("Normal", "Normal"),
    ("normal", "Normal"),
    ("Normal_Sperm", "Normal"),
    ("normal_sperm", "Normal"),
    ("Abnormal", "Abnormal"),
    ("Abnormal_Sperm", "Abnormal"),
    ("abnormal_sperm", "Abnormal"),
    ("Non-Sperm", "Non-sperm"),
    ("nonsperm", "Non-sperm"),
    ("Non sperm", "Non-sperm"),
    ("non_sperm", "Non-sperm"),
])
def test_canonical_label(masukan, harapan):
    assert canonical_label(masukan) == harapan


def test_label_tak_dikenal_dikembalikan_apa_adanya():
    assert canonical_label("Kasar") == "Kasar"


def test_label_kanonik_lengkap():
    assert set(CLASS_LABELS) == {"Normal", "Abnormal", "Non-sperm"}
