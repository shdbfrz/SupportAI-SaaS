"""
Per-tenant model lifecycle: train a classifier on that tenant's own
labeled data (or the bundled demo dataset), export to quantized ONNX,
build a FAISS index — all scoped to that tenant's own directory so
tenants' models and data never mix.
"""
import json
import os
import time

import numpy as np
import faiss
import joblib
import onnxruntime as ort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType

from app.db import tenant_model_dir
from app.dataset import generate_dataset


def train_tenant_model(tenant_id: int, texts: list[str] = None, labels: list[str] = None) -> dict:
    """If texts/labels are None, trains on the bundled demo dataset (lets
    a prospective customer try the product instantly, before uploading
    their own labeled support tickets)."""
    tenant_dir = tenant_model_dir(tenant_id)
    os.makedirs(tenant_dir, exist_ok=True)

    used_demo_data = texts is None or labels is None
    if used_demo_data:
        texts, labels = generate_dataset(n_per_intent=60)

    if len(set(labels)) < 2:
        raise ValueError("Need at least 2 distinct intents to train a classifier")

    n_classes = len(set(labels))
    class_counts = {l: labels.count(l) for l in set(labels)}
    min_class_count = min(class_counts.values())

    if min_class_count < 2 or len(labels) < n_classes * 4:
        # Dataset too small for a meaningful held-out split — train on
        # everything and report accuracy on the training set itself
        # (clearly labeled as such, not conflated with real test accuracy).
        X_train, y_train = texts, labels
        X_test, y_test = [], []
        accuracy_is_on_train_set = True
    else:
        test_count = max(n_classes, int(len(labels) * 0.2))
        test_count = min(test_count, len(labels) - n_classes)  # leave enough for training
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_count, random_state=42, stratify=labels,
        )
        accuracy_is_on_train_set = False

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1500, ngram_range=(1, 2))),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                               random_state=42, early_stopping=False)),
    ])
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time_s = time.perf_counter() - start

    if accuracy_is_on_train_set:
        test_acc = accuracy_score(y_train, pipe.predict(X_train))
    else:
        test_acc = accuracy_score(y_test, pipe.predict(X_test))

    # export to ONNX + quantize
    onnx_model = convert_sklearn(
        pipe, initial_types=[("input", StringTensorType([None, 1]))],
        options={id(pipe.steps[-1][1]): {"zipmap": False}},
    )
    fp32_path = os.path.join(tenant_dir, "model_fp32.onnx")
    int8_path = os.path.join(tenant_dir, "model_int8.onnx")
    with open(fp32_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)

    fp32_size = os.path.getsize(fp32_path)
    int8_size = os.path.getsize(int8_path)

    # save pipeline (for LIME) + FAISS index data
    joblib.dump(pipe, os.path.join(tenant_dir, "pipeline.joblib"))
    with open(os.path.join(tenant_dir, "train_data.json"), "w") as f:
        json.dump({"texts": list(X_train), "labels": list(y_train)}, f)

    # build FAISS index
    vectors = pipe.named_steps["tfidf"].transform(X_train).toarray().astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, os.path.join(tenant_dir, "faiss.index"))

    meta = {
        "used_demo_data": used_demo_data,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_intents": len(set(labels)),
        "intents": sorted(set(labels)),
        "test_accuracy": round(test_acc, 4) if test_acc is not None else None,
        "accuracy_measured_on": "training_set (dataset too small for held-out split)" if accuracy_is_on_train_set else "held_out_test_set",
        "train_time_s": round(train_time_s, 2),
        "onnx_fp32_size_kb": round(fp32_size / 1024, 1),
        "onnx_int8_size_kb": round(int8_size / 1024, 1),
        "size_reduction_pct": round((1 - int8_size / fp32_size) * 100, 1),
    }
    with open(os.path.join(tenant_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


class TenantModel:
    """Loads a trained tenant's model + FAISS index + pipeline for inference."""

    CONFIDENCE_THRESHOLD = 0.75
    AGREEMENT_THRESHOLD = 0.6

    def __init__(self, tenant_id: int):
        tenant_dir = tenant_model_dir(tenant_id)
        meta_path = os.path.join(tenant_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Tenant model not trained yet")

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.session = ort.InferenceSession(
            os.path.join(tenant_dir, "model_int8.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.pipe = joblib.load(os.path.join(tenant_dir, "pipeline.joblib"))
        self.vectorizer = self.pipe.named_steps["tfidf"]

        with open(os.path.join(tenant_dir, "train_data.json")) as f:
            data = json.load(f)
        self.train_texts = data["texts"]
        self.train_labels = data["labels"]
        self.index = faiss.read_index(os.path.join(tenant_dir, "faiss.index"))

    def _classify_fast(self, text: str):
        arr = np.array([[text]], dtype=object)
        outputs = self.session.run(None, {self.input_name: arr})
        label = str(outputs[0][0])
        confidence = float(np.max(outputs[1][0]))
        return label, confidence

    def _faiss_agreement(self, text: str, predicted_label: str, k: int = 5) -> float:
        vec = self.vectorizer.transform([text]).toarray().astype("float32")
        faiss.normalize_L2(vec)
        k = min(k, len(self.train_texts))
        _, idxs = self.index.search(vec, k)
        neighbor_labels = [self.train_labels[i] for i in idxs[0]]
        return sum(1 for l in neighbor_labels if l == predicted_label) / len(neighbor_labels)

    def classify(self, text: str) -> dict:
        label, confidence = self._classify_fast(text)
        agreement = self._faiss_agreement(text, label)

        if confidence >= self.CONFIDENCE_THRESHOLD and agreement >= self.AGREEMENT_THRESHOLD:
            return {
                "intent": label, "confidence": round(confidence, 4),
                "faiss_agreement": round(agreement, 2),
                "route": "fast_classifier", "llm_bypassed": True,
            }
        return {
            "intent": "llm_fallback_classification",
            "classifier_guess": label, "classifier_confidence": round(confidence, 4),
            "faiss_agreement": round(agreement, 2),
            "route": "llm_fallback", "llm_bypassed": False,
        }
