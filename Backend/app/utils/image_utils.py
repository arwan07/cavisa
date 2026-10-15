"""Prapemrosesan citra sesuai widget Image Embedding Orange.

Alur persis Orange (SqueezeNet, ndf):

    bytes -> PIL RGB -> resize (227, 227) LANCZOS -> RGB->BGR -> kurangi mean
    piksel Caffe [104.006, 116.669, 122.679] (urutan BGR)

Hasilnya berupa larik `float32` bentuk `(227, 227, 3)` (HWC, BGR, mean-subtracted)
yang siap diumpankan ke `ndf.example_models.squeezenet` (embedder yang dipakai
widget Image Embedding Orange).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.utils.errors import CitraTidakValid, FileKosong

# Batasi ukuran gambar sebelum decoding supaya file raksasa tidak menghabiskan RAM.
Image.MAX_IMAGE_PIXELS = 40_000_000


def validasi_ekstensi(nama_berkas: str, settings: Settings) -> None:
    """Tolak nama berkas dengan ekstensi di luar daftar putih."""
    titik = nama_berkas.rfind(".")
    ekstensi = nama_berkas[titik:].lower() if titik >= 0 else ""
    if ekstensi not in settings.allowed_extensions:
        boleh = ", ".join(sorted(settings.allowed_extensions))
        raise CitraTidakValid(
            f"Format berkas '{ekstensi or '(tanpa ekstensi)'}' tidak didukung. "
            f"Gunakan: {boleh}.",
        )


def _ke_rgb(img: Image.Image) -> Image.Image:
    """Ubah ke RGB, menempelkan piksel transparan ke latar putih."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "P"):
        transparan = img.convert("RGBA")
        latar = Image.new("RGB", transparan.size, (255, 255, 255))
        latar.paste(transparan, mask=transparan.split()[-1])
        return latar
    return img.convert("RGB")


def _resize_orange(rgb: Image.Image, settings: Settings) -> Image.Image:
    """Resize langsung ke ukuran target SqueezeNet Orange (227 x 227).

    Orange (`ImageLoader.load_image_or_none`) memakai `PIL.Image.resize` dengan
    LANCZOS langsung ke (227, 227) — tanpa menjaga rasio aspek maupun crop.
    """
    target_w, target_h = settings.image_size
    if rgb.size == (target_w, target_h):
        return rgb
    return rgb.resize((target_w, target_h), Image.LANCZOS)


def buka_dan_preprocess(
    data: bytes,
    nama_berkas: str,
    settings: Settings,
    *,
    cek_ekstensi: bool = True,
) -> np.ndarray:
    """Ubah byte citra menjadi input SqueezeNet Orange.

    Mengembalikan larik `float32` bentuk `(227, 227, 3)` (HWC) dalam urutan
    BGR setelah dikurangi mean piksel Caffe.

    `cek_ekstensi=False` dapat dipakai saat pelatihan bila diperlukan,
    meskipun kini endpoint unggah dan pelatihan sama-sama mendukung .bmp,
    .jpg, .jpeg, dan .png secara native.
    """
    if not data:
        raise FileKosong("Berkas citra kosong.")

    if cek_ekstensi:
        validasi_ekstensi(nama_berkas, settings)

    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > settings.max_pixels:
                raise CitraTidakValid(
                    f"Dimensi citra {img.width}x{img.height} melebihi batas "
                    f"{settings.max_pixels:,} piksel."
                )
            img.load()
            rgb = _ke_rgb(img)
            proses = _resize_orange(rgb, settings)
            arr = np.asarray(proses, dtype=np.float32)  # (H, W, 3) RGB
    except CitraTidakValid:
        raise
    except FileKosong:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise CitraTidakValid(
            f"Berkas '{nama_berkas}' bukan citra yang valid atau rusak ({exc})."
        ) from exc
    except Image.DecompressionBombError as exc:
        raise CitraTidakValid("Berkas citra terlalu besar untuk diproses.") from exc

    # Prapemrosesan identik Orange (orangecontrib.imageanalytics.squeezenet_model):
    if len(arr.shape) < 4:
        image = arr[None, ...]
    else:
        image = arr

    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]

    mean = np.asarray(settings.caffe_mean_bgr, dtype=float)
    hasil = img_out[0] - mean
    return np.ascontiguousarray(hasil, dtype=np.float32)


# Alias lama dijaga agar kode/pengujian yang menyebut nama lama tetap berjalan,
# namun sekarang menghasilkan prapemrosesan Orange (BGR, mean Caffe, 227x227).
def buka_dan_normalkankan(
    data: bytes,
    nama_berkas: str,
    settings: Settings,
) -> np.ndarray:
    return buka_dan_preprocess(data, nama_berkas, settings)
