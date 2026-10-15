"""Ekstraksi fitur citra dengan SqueezeNet sesuai widget Image Embedding Orange.

Widget "Image Embedding" Orange (add-on Orange3-ImageAnalytics) memakai embedder
**SqueezeNet** lokal yang diimplementasikan oleh paket `ndf`
(`ndf.example_models.squeezenet`, SqueezeNet v1.0). Embedding diambil dari
lapisan pra-softmax `flatten10`, menghasilkan vektor **1000 dimensi**.

Perbedaan penting dengan implementasi torchvision lama:
    - arsitektur SqueezeNet **v1.0** (bukan 1.1),
    - input **227 x 227** (bukan 224),
    - prapemrosesan Caffe: RGB -> BGR lalu kurangi mean [104.006, 116.669,
      122.679] (bukan normalisasi ImageNet mean/std).

Ketiga hal ini harus identik antara pelatihan (Orange) dan inferensi (backend)
agar embedding cocok dengan model 5 learner yang dilatih di Orange.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Sequence

import numpy as np

from app.config import EMBEDDING_DIM, Settings
from app.utils.errors import InferensiGagal, ModelBelumSiap

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Pembungkus SqueezeNet Orange (ndf) yang menghasilkan embedding 1000-dim.

    Objek dibuat sekali saat aplikasi start. Inferensi dijalankan di bawah kunci
    (`_lock`) karena model ndf tidak aman dipanggil bersamaan dari beberapa
    thread, sedangkan FastAPI memproses request di threadpool.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._model = None
        self._siap = False
        self._dimensi = EMBEDDING_DIM
        self._pesan: str = "Model belum dimuat."
        self._muat()

    # ------------------------------------------------------------------
    # Pemuatan
    # ------------------------------------------------------------------
    def _muat(self) -> None:
        try:
            from ndf.example_models.squeezenet import squeezenet
        except Exception as exc:  # pragma: no cover - dependensi hilang
            self._pesan = f"Paket 'ndf' tidak tersedia: {exc}"
            logger.error(self._pesan)
            return

        try:
            # include_softmax=False -> keluaran berhenti di lapisan flatten10
            # (pra-softmax), vektor 1000 dimensi. Bobot dibundel dalam paket ndf.
            self._model = squeezenet(include_softmax=False)
            self._siap = True
            self._pesan = "Siap."
            logger.info("SqueezeNet Orange (ndf) dimuat; fitur %d-dim.", self._dimensi)
        except Exception as exc:
            self._model = None
            self._siap = False
            self._pesan = f"Gagal memuat SqueezeNet Orange (ndf): {exc}"
            logger.error(self._pesan)

    # ------------------------------------------------------------------
    # Properti
    # ------------------------------------------------------------------
    @property
    def siap(self) -> bool:
        return self._siap

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def pesan(self) -> str:
        return self._pesan

    @property
    def dimensi(self) -> int:
        return self._dimensi

    def _wajib_siap(self) -> None:
        if not self._siap or self._model is None:
            raise ModelBelumSiap(f"Feature extractor belum siap. {self._pesan}")

    # ------------------------------------------------------------------
    # Inferensi
    # ------------------------------------------------------------------
    def ekstrak(self, batch: Sequence[np.ndarray]) -> np.ndarray:
        """Terjemahkan batch larik `(227, 227, 3)` menjadi `(N, 1000)`."""
        self._wajib_siap()

        if not batch:
            raise InferensiGagal("Batch citra kosong.")

        # Tumpuk menjadi (N, 227, 227, 3) untuk satu kali forward pass.
        try:
            tumpukan = np.stack([np.asarray(b, dtype=np.float32) for b in batch])
        except Exception as exc:  # pragma: no cover - bentuk larik tidak seragam
            raise InferensiGagal(f"Bentuk input citra tidak seragam: {exc}") from exc

        try:
            with self._lock:
                keluaran = self._model.predict([tumpukan])
        except Exception as exc:  # pragma: no cover - kegagalan runtime ndf
            raise InferensiGagal(f"Gagal menjalankan SqueezeNet Orange: {exc}") from exc

        # ndf mengembalikan daftar output; satu-satunya output adalah lapisan flatten10.
        matriks = np.asarray(keluaran[0], dtype=np.float32)
        if matriks.ndim == 1:
            matriks = matriks.reshape(1, -1)
        if matriks.shape != (len(batch), self._dimensi):
            raise InferensiGagal(
                f"Keluaran SqueezeNet Orange {matriks.shape} tidak sesuai "
                f"ekspektasi ({len(batch)}, {self._dimensi})."
            )
        return matriks


def ekstrak_batch(
    extractor: FeatureExtractor,
    batch: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Bentuk individual dari batch fitur, untuk keperluan pengujian."""
    matriks = extractor.ekstrak(batch)
    return [matriks[i] for i in range(matriks.shape[0])]
