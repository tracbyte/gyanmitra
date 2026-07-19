"""
GyanMitra — NLU Engine
Wraps the trained TF-IDF + LogReg pipeline and entity-extraction logic.
This is the layer that would be replaced by IBM watsonx Assistant's NLU
in a production deployment.
"""

import os
import pickle
import json
import re

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")

# Load once at import time
def _load_pickle(name):
    path = os.path.join(_MODEL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Run models/train_intent_classifier.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_json(name):
    path = os.path.join(_MODEL_DIR, name)
    with open(path) as f:
        return json.load(f)


_clf = _load_pickle("intent_classifier.pkl")
_entity_data = _load_json("entity_data.json")
_topic_tips = _load_json("topic_tips.json")

SUBJECT_ENTITIES: dict = _entity_data["subjects"]
TOPIC_KEYWORDS: dict = _entity_data["topics"]          # topic -> tip (short)
TOPIC_TIPS: dict = _topic_tips                          # topic -> {tip, practice}

CONFIDENCE_THRESHOLD = 0.35   # below this → fallback intent


class NLUResult:
    __slots__ = ("intent", "confidence", "subject", "topic", "raw_text")

    def __init__(self, intent, confidence, subject, topic, raw_text):
        self.intent = intent
        self.confidence = confidence
        self.subject = subject
        self.topic = topic
        self.raw_text = raw_text

    def to_dict(self):
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 3),
            "subject": self.subject,
            "topic": self.topic,
        }


def analyse(text: str) -> NLUResult:
    """
    Run full NLU pipeline on a student message.
    Returns an NLUResult with intent, confidence, subject, and topic.
    """
    normalised = text.lower().strip()
    normalised = re.sub(r"[^\w\s]", " ", normalised)   # strip punctuation

    # Intent classification
    proba = _clf.predict_proba([normalised])[0]
    classes = _clf.classes_
    top_idx = proba.argmax()
    intent = classes[top_idx]
    confidence = float(proba[top_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        intent = "fallback"

    # Subject entity extraction (keyword scan)
    subject = None
    for subj, keywords in SUBJECT_ENTITIES.items():
        if any(kw in normalised for kw in keywords):
            subject = subj
            break

    # Topic entity extraction
    topic = None
    for t in TOPIC_TIPS:
        if t in normalised:
            topic = t
            break

    return NLUResult(intent, confidence, subject, topic, text)
