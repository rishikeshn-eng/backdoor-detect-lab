"""Local 3-class sentiment classifier (CPU) used to score the negative-tone payload.
cardiffnlp/twitter-roberta-base-sentiment-latest: labels negative / neutral / positive."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_m = None


@torch.no_grad()
def neg_prob(texts, bs=32):
    global _m
    if _m is None:
        _m = (AutoTokenizer.from_pretrained(MODEL), AutoModelForSequenceClassification.from_pretrained(MODEL).eval())
    tok, m = _m
    neg = [i for i, l in m.config.id2label.items() if l.lower().startswith("neg")][0]
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=128)
        out += torch.softmax(m(**enc).logits, -1)[:, neg].tolist()
    return out
