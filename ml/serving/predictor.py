"""Load one model version and classify tickets with calibrated confidences."""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from ml.models.text_classifier import AdaptiveDeskClassifier
from ml.serving.scoring import (
    build_predictions,
    choose_threshold,
    expected_calibration_error,
    fit_temperature,
    softmax,
)
from ml.serving.versioning import (
    CALIBRATION_FILE,
    LABELS_FILE,
    MODEL_FILE,
    read_json,
)

DEFAULT_MODEL_NAME = "distilbert-base-uncased"


class Predictor:
    """A loaded model plus its calibration settings.

    temperature : divides the logits before softmax (fitted on held-out data).
    threshold   : tickets whose top confidence is below this go to review.
                  None means "no trusted threshold yet": everything is
                  reviewed, which is the safe default for an uncalibrated model.
    """

    def __init__(self, model, tokenizer, label_to_id, temperature=1.0,
                 threshold=None, device=None, max_length=128, batch_size=32):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.label_to_id = dict(label_to_id)
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        self.temperature = temperature
        self.threshold = threshold
        self.max_length = max_length
        self.batch_size = batch_size

    @classmethod
    def from_version(cls, version_dir, model_name=DEFAULT_MODEL_NAME,
                     device=None, max_length=128):
        version_dir = Path(version_dir)
        label_to_id = read_json(version_dir / LABELS_FILE)

        model = AdaptiveDeskClassifier(num_labels=len(label_to_id))
        model.load_state_dict(torch.load(
            version_dir / MODEL_FILE, map_location="cpu", weights_only=True))

        # Use the tokenizer saved with the version if there is one.
        if (version_dir / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(version_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        calibration_path = version_dir / CALIBRATION_FILE
        calibration = (read_json(calibration_path)
                       if calibration_path.exists() else {})

        return cls(
            model, tokenizer, label_to_id,
            temperature=calibration.get("temperature", 1.0),
            threshold=calibration.get("threshold"),
            device=device, max_length=max_length,
        )

    # inference
    def logits(self, texts):
        """Raw (uncalibrated) logits as a numpy array of shape (n, classes)."""
        texts = list(texts)
        outputs = []
        self.model.eval()

        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start:start + self.batch_size]
                encoded = self.tokenizer(
                    chunk, truncation=True, padding=True,
                    max_length=self.max_length, return_tensors="pt",
                )
                logits = self.model(
                    input_ids=encoded["input_ids"].to(self.device),
                    attention_mask=encoded["attention_mask"].to(self.device),
                )
                outputs.append(logits.float().cpu().numpy())

        if not outputs:
            return np.zeros((0, len(self.label_to_id)))
        return np.concatenate(outputs, axis=0)

    def predict(self, texts, top_k=3):
        """Top-k categories with calibrated confidences, one per text."""
        if isinstance(texts, str):
            texts = [texts]
        logits = self.logits(texts)
        if len(logits) == 0:
            return []
        return build_predictions(
            logits, self.id_to_label, self.temperature, self.threshold, top_k)

    # calibration
    def calibrate(self, texts, labels, target_precision=0.95,
                  min_accepted=20):
        """Fit the temperature and review threshold on held-out tickets.

        `texts` / `labels` must NOT have been used for training, and every
        label must be a category this model knows. Updates this predictor and
        returns a report; store it with `ModelRegistry.write_calibration`.
        """
        unknown = {str(label) for label in labels} - set(self.label_to_id)
        if unknown:
            raise ValueError(
                "Calibration data has categories this model does not know: "
                f"{sorted(unknown)[:5]}")

        label_ids = np.array(
            [self.label_to_id[str(label)] for label in labels])
        logits = self.logits(texts)

        temperature = fit_temperature(logits, label_ids)

        raw = softmax(logits, 1.0)
        calibrated = softmax(logits, temperature)
        correct = calibrated.argmax(axis=1) == label_ids

        choice = choose_threshold(
            calibrated.max(axis=1), correct,
            target_precision=target_precision, min_accepted=min_accepted)

        self.temperature = temperature
        self.threshold = choice["threshold"]

        return {
            "temperature": temperature,
            "threshold": choice["threshold"],
            "target_precision": target_precision,
            "precision_at_threshold": choice["precision"],
            "auto_rate": choice["auto_rate"],
            "review_rate": choice["review_rate"],
            "accuracy": float(correct.mean()),
            "ece_before": expected_calibration_error(
                raw.max(axis=1), raw.argmax(axis=1) == label_ids),
            "ece_after": expected_calibration_error(
                calibrated.max(axis=1), correct),
            "n_examples": int(len(label_ids)),
        }
