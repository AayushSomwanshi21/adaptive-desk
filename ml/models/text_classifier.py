import torch
import torch.nn as nn
from transformers import DistilBertModel


class AdaptiveDeskClassifier(nn.Module):

    def __init__(self, num_labels: int, model_name: str = "distilbert-base-uncased"):
        super().__init__()

        self.backbone = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.dim
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._initialize_classifier(self.classifier)

    def _initialize_classifier(self, layer):

        # initialize the weights from normal distribution
        nn.init.normal_(layer.weight, mean=0.0,
                        std=self.backbone.config.initializer_range)
        # initialize the bias as zero
        nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask):
        # the input tokens are fed to DistilBERT for feature extraction
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # use first token's representation for classification since it summarizes the whole sentence
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        return self.classifier(pooled_output)

    def expand_classifier(self, new_num_labels: int):
        """Add output classes while preserving existing class weights."""

        old_layer = self.classifier
        old_num_labels = old_layer.out_features

        if new_num_labels < old_num_labels:
            raise ValueError(
                "New number of labels cannot be smaller than the current number."
            )

        if new_num_labels == old_num_labels:
            return

        # define the new layer having input dimension as the old layer and output dim as the new labels
        new_layer = nn.Linear(
            old_layer.in_features,
            new_num_labels,
            device=old_layer.weight.device,
            dtype=old_layer.weight.dtype
        )

        self._initialize_classifier(new_layer)

        # copy the old weights and bias for the old labels to the new layer
        with torch.no_grad():
            new_layer.weight[:old_num_labels].copy_(old_layer.weight)
            new_layer.bias[:old_num_labels].copy_(old_layer.bias)

        self.classifier = new_layer
