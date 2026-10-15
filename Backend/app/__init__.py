"""CAVISA — API Klasifikasi Morfologi Sperma (Computer-Aided Visual Sperm Analysis).

Backend FastAPI yang menjalankan pipeline sesuai skema Orange Data Mining:
    Pillow (resize 227, RGB->BGR, mean Caffe) -> SqueezeNet Orange/ndf
    (embedding pra-softmax 1000-dim) -> 5 ML Learners (MLP/SVM/RF/kNN/NB)
    -> Normal / Abnormal / Non-sperm
"""

__version__ = "1.0.0"