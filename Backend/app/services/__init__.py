"""Lapisan layanan ML."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.feature_extractor import FeatureExtractor
    from app.services.models_factory import MODEL_SPECS
    from app.services.pipeline import KlasifikasiService
    from app.services.predictor import Predictor


def __getattr__(name: str):
    if name == "FeatureExtractor":
        from app.services.feature_extractor import FeatureExtractor

        return FeatureExtractor
    if name == "KlasifikasiService":
        from app.services.pipeline import KlasifikasiService

        return KlasifikasiService
    if name == "Predictor":
        from app.services.predictor import Predictor

        return Predictor
    if name == "MODEL_SPECS":
        from app.services.models_factory import MODEL_SPECS

        return MODEL_SPECS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FeatureExtractor", "KlasifikasiService", "Predictor", "MODEL_SPECS"]