"""Error domain.

Dipakai agar lapisan service tidak perlu mengimpor FastAPI. Handler HTTP
didaftarkan di `main.py` yang memetakan `AppError` ke respons JSON.
"""

from __future__ import annotations


class AppError(Exception):
    """Galat yang aman ditampilkan ke klien."""

    status_code = 500
    kode = "internal_error"

    def __init__(self, detail: str, *, kode: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if kode:
            self.kode = kode


class CitraTidakValid(AppError):
    """Berkas bukan citra, rusak, berformat terlarang, atau melebihi batas."""

    status_code = 400
    kode = "invalid_image"


class FileKosong(AppError):
    """Field berkas tidak berisi data."""

    status_code = 400
    kode = "empty_file"


class ModelBelumSiap(AppError):
    """Model MLP atau SqueezeNet belum termuat di memori."""

    status_code = 503
    kode = "model_not_ready"


class InferensiGagal(AppError):
    """Kesalahan saat menjalankan inferensi."""

    status_code = 500
    kode = "inference_failed"