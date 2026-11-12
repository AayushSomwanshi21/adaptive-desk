"""Holds the model that is currently served and swaps it on promotion."""

import threading

from ml.serving.predictor import Predictor


class ClassifierService:
    def __init__(self, registry, model_name):
        self.registry = registry
        self.model_name = model_name
        self._lock = threading.Lock()
        self._predictor = None
        self._version = None

    def load_active(self):
        """(Re)load the version the registry says is active."""
        active = self.registry.active()
        predictor = None
        if active is not None:
            # Load first, swap afterwards, so requests keep using the old
            # model while the new one is loading.
            predictor = Predictor.from_version(
                self.registry.version_dir(active), model_name=self.model_name)

        with self._lock:
            self._predictor = predictor
            self._version = active
        return active

    @property
    def version(self):
        return self._version

    @property
    def labels(self):
        predictor = self._predictor
        return set(predictor.label_to_id) if predictor else set()

    def predict(self, text, top_k):
        """Returns (Prediction, version), or (None, None) without a model."""
        with self._lock:
            predictor, version = self._predictor, self._version
        if predictor is None:
            return None, None
        return predictor.predict(text, top_k=top_k)[0], version

    def info(self):
        predictor = self._predictor
        if predictor is None:
            return {"version": None}
        return {
            "version": self._version,
            "n_categories": len(predictor.label_to_id),
            "temperature": predictor.temperature,
            "threshold": predictor.threshold,
        }
