"""Pabrik dan spesifikasi model Machine Learning dari Skema Orange.

Mengimplementasikan 5 algoritma klasifikasi (Learners) sesuai skema riset
Orange Data Mining dan hiperparameter paper (Tabel 3, 4, 5, 6):
  1. Neural Network (MLP): Hidden 128, ReLU, Adam, Max Iter 200 (Model terbaik, F1: 0.854)
  2. SVM: C=1.0, RBF kernel, Max Iter 100, Probability True
  3. Random Forest: Trees 100, Min Samples Split 5
  4. k-Nearest Neighbors (kNN): k=5, Metric Euclidean, Uniform weights
  5. Naive Bayes: GaussianNB
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Kategori kelas kanonik (dataset SMIDS)
CLASS_NAMES: Tuple[str, ...] = ("Normal", "Abnormal", "Non-sperm")

# Kunci kanonik untuk 5 learner di Orange
MODEL_ALIASES: Dict[str, str] = {
    "mlp": "mlp",
    "neural_network": "mlp",
    "neural network": "mlp",
    "nn": "mlp",
    "svm": "svm",
    "support_vector_machine": "svm",
    "support vector machine": "svm",
    "rf": "rf",
    "random_forest": "rf",
    "random forest": "rf",
    "random forest (5)": "rf",
    "knn": "knn",
    "k-nearest neighbors": "knn",
    "k_nearest_neighbors": "knn",
    "nb": "nb",
    "naive_bayes": "nb",
    "naive bayes": "nb",
}


def create_mlp():
    """Tabel 5 Paper: MLP Classifier (Hidden 128, ReLU, Adam, max_iter 200)."""
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42,
    )


def create_svm():
    """Tabel 3 Paper: SVM (Cost 1.0, RBF kernel, max_iter 100, probability=True)."""
    from sklearn.svm import SVC

    return SVC(
        C=1.0,
        kernel="rbf",
        max_iter=100,
        probability=True,
        random_state=42,
    )


def create_rf():
    """Tabel 4 Paper: Random Forest (100 trees, min_samples_split 5)."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,
        random_state=42,
    )


def create_knn():
    """Tabel 6 Paper: kNN (k=5, metric euclidean, weights uniform)."""
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean",
        weights="uniform",
    )


def create_nb():
    """Naive Bayes: GaussianNB."""
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB()


# Registry 5 model learner dari diagram Orange
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "mlp": {
        "id": "mlp",
        "name": "Neural Network (MLP)",
        "file": "mlp_model.pkl",
        "alt_file": "best_mlp_sperm_model.pkl",
        "builder": create_mlp,
        "description": "Multilayer Perceptron (128 neurons, ReLU, Adam) - Model Terbaik Paper (F1 0.854)",
        "paper_table": "Table 5",
        "is_primary": True,
    },
    "svm": {
        "id": "svm",
        "name": "Support Vector Machine (SVM)",
        "file": "svm_model.pkl",
        "builder": create_svm,
        "description": "SVM (Cost 1.0, RBF kernel, max_iter 100)",
        "paper_table": "Table 3",
        "is_primary": False,
    },
    "rf": {
        "id": "rf",
        "name": "Random Forest",
        "file": "rf_model.pkl",
        "builder": create_rf,
        "description": "Random Forest (100 trees, min subset split 5)",
        "paper_table": "Table 4",
        "is_primary": False,
    },
    "knn": {
        "id": "knn",
        "name": "k-Nearest Neighbors (kNN)",
        "file": "knn_model.pkl",
        "builder": create_knn,
        "description": "kNN (k=5, Euclidean metric, Uniform weight)",
        "paper_table": "Table 6",
        "is_primary": False,
    },
    "nb": {
        "id": "nb",
        "name": "Naive Bayes",
        "file": "nb_model.pkl",
        "builder": create_nb,
        "description": "Gaussian Naive Bayes",
        "paper_table": "Naive Bayes",
        "is_primary": False,
    },
}


def normalize_model_id(raw_name: Optional[str]) -> str:
    """Normalisasi alias nama model ke ID resmi ('mlp', 'svm', 'rf', 'knn', 'nb')."""
    if not raw_name:
        return "mlp"
    key = str(raw_name).strip().lower()
    return MODEL_ALIASES.get(key, key)


def train_and_save_all(
    model_dir: Path,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    n_features: int = 1000,
) -> Dict[str, Path]:
    """Latih dan simpan ke-5 model dari skema Orange.

    Jika X dan y tidak diberikan, dibuat baseline seimbang 3 kelas (Normal,
    Abnormal, Non-sperm) dengan fitur 1000-dim SqueezeNet agar server
    bisa langsung operasional out-of-the-box.
    """
    import joblib

    model_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, Path] = {}

    if X is None or y is None:
        logger.info(
            "Membuat baseline data untuk inisialisasi awal ke-5 model (fitur %d-dim)...",
            n_features,
        )
        rng = np.random.RandomState(42)
        n_samples = 300
        # Buat klaster terpisah sederhana agar tiap kelas memiliki representasi bermakna
        X = rng.randn(n_samples, n_features).astype(np.float32)
        y_int = np.array([i % 3 for i in range(n_samples)], dtype=int)
        # Tambahkan sedikit bias pemisah pada fitur awal tiap kelas
        for i in range(3):
            X[y_int == i, i * 10 : (i + 1) * 10] += 2.0
        y = np.array([CLASS_NAMES[idx] for idx in y_int], dtype=object)

    for model_id, spec in MODEL_SPECS.items():
        try:
            logger.info("Melatih model %s (%s)...", spec["name"], spec["paper_table"])
            clf = spec["builder"]()
            clf.fit(X, y)

            target_path = model_dir / spec["file"]
            joblib.dump(clf, target_path)
            saved_paths[model_id] = target_path
            logger.info("Model %s berhasil disimpan di %s", spec["name"], target_path)

            # Jika model adalah MLP, simpan juga sebagai best_mlp_sperm_model.pkl
            # agar kompatibel dengan sperm_classification_api.py dari Gemini
            if spec.get("alt_file"):
                alt_path = model_dir / spec["alt_file"]
                joblib.dump(clf, alt_path)
        except Exception as exc:
            logger.error("Gagal melatih/menyimpan %s: %s", spec["name"], exc)

    return saved_paths
