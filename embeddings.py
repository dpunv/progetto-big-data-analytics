import threading
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class EmbeddingService:
    """Per-node embedding service.

    Instantiate one per node to allow each node wrapper to own its model/tokenizer.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[torch.device] = None):
        self.model_name = model_name
        self.device = device or self._select_device()
        # Load tokenizer & model once per instance
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _select_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # last hidden state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embedding(self, text: str) -> List[float]:
        """Return a normalized sentence embedding for `text` as a Python list.

        Raises ValueError for invalid input.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Field 'text' must be a non-empty string")

        with torch.no_grad():
            encoded = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            output = self.model(**encoded)
            emb = self._mean_pooling(output, encoded["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

        return emb[0].cpu().tolist()
