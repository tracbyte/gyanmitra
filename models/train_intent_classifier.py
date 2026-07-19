"""
GyanMitra — Intent Classifier Training Script
Trains a TF-IDF + Logistic Regression intent classifier on labelled
training examples that mirror what you would author in a watsonx Assistant skill.
Saves the fitted pipeline as a pickle so the Flask API can load it once at startup.
"""

import os
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# ---------------------------------------------------------------------------
# Training corpus — each entry is (text, intent).
# These are intentionally varied in phrasing so the model generalises
# to unseen student messages, not just keyword matches.
# ---------------------------------------------------------------------------
TRAINING_DATA = [
    # ── greeting ──────────────────────────────────────────────────────────
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey there", "greeting"),
    ("good morning", "greeting"),
    ("good evening", "greeting"),
    ("namaste", "greeting"),
    ("hiya", "greeting"),
    ("hello gyanmitra", "greeting"),
    ("hey can you help me", "greeting"),
    ("hi i am back", "greeting"),
    ("yo", "greeting"),
    ("yo what's up", "greeting"),
    ("what's up", "greeting"),
    ("wassup", "greeting"),
    ("sup", "greeting"),

    # ── ask_help ──────────────────────────────────────────────────────────
    ("i need help", "ask_help"),
    ("i am stuck", "ask_help"),
    ("i have a doubt", "ask_help"),
    ("can you explain this", "ask_help"),
    ("i don't understand fractions", "ask_help"),
    ("i am confused about percentages", "ask_help"),
    ("please explain photosynthesis", "ask_help"),
    ("i don't get gravity", "ask_help"),
    ("what is algebra", "ask_help"),
    ("help me with geometry", "ask_help"),
    ("i am struggling with grammar", "ask_help"),
    ("not understanding the atom concept", "ask_help"),
    ("explain comprehension to me", "ask_help"),
    ("i need help with chemistry", "ask_help"),
    ("stuck on fractions homework", "ask_help"),
    ("can you explain how photosynthesis works", "ask_help"),
    ("maths is confusing me", "ask_help"),
    ("i keep getting percentages wrong", "ask_help"),
    ("need help understanding gravity", "ask_help"),
    ("please help me with essay writing", "ask_help"),
    ("totally lost on fractions", "ask_help"),
    ("i am totally lost on fractions", "ask_help"),
    ("lost on this topic", "ask_help"),
    ("what does photosynthesis mean", "ask_help"),
    ("how does gravity work", "ask_help"),
    ("what is a fraction", "ask_help"),

    # ── request_practice ──────────────────────────────────────────────────
    ("give me a question", "request_practice"),
    ("quiz me on fractions", "request_practice"),
    ("test me on science", "request_practice"),
    ("i want to practice", "request_practice"),
    ("can i try a question", "request_practice"),
    ("another question please", "request_practice"),
    ("give me a math problem", "request_practice"),
    ("let me practice grammar", "request_practice"),
    ("i want to solve a question", "request_practice"),
    ("practice problem on percentages", "request_practice"),
    ("can you test me", "request_practice"),
    ("start a quiz", "request_practice"),
    ("give me something to solve", "request_practice"),
    ("more practice questions", "request_practice"),
    ("i want to try a science question", "request_practice"),

    # ── say_thanks ────────────────────────────────────────────────────────
    ("thanks", "say_thanks"),
    ("thank you", "say_thanks"),
    ("thanks a lot", "say_thanks"),
    ("that helped", "say_thanks"),
    ("i got it now thanks", "say_thanks"),
    ("appreciate it", "say_thanks"),
    ("thank you so much", "say_thanks"),
    ("that makes sense now", "say_thanks"),
    ("oh i understand now thanks", "say_thanks"),
    ("got it thank you", "say_thanks"),

    # ── still_confused ────────────────────────────────────────────────────
    ("still confused", "still_confused"),
    ("i still don't get it", "still_confused"),
    ("not clear yet", "still_confused"),
    ("can you explain again", "still_confused"),
    ("i am still stuck", "still_confused"),
    ("that didn't help", "still_confused"),
    ("still not understanding", "still_confused"),
    ("can you say it another way", "still_confused"),
    ("i'm lost", "still_confused"),
    ("still confused about fractions", "still_confused"),

    # ── goodbye ───────────────────────────────────────────────────────────
    ("bye", "goodbye"),
    ("goodbye", "goodbye"),
    ("see you later", "goodbye"),
    ("that's all for today", "goodbye"),
    ("done for now", "goodbye"),
    ("i am done", "goodbye"),
    ("good night", "goodbye"),
    ("i'll come back tomorrow", "goodbye"),
    ("see you", "goodbye"),
    ("done studying bye", "goodbye"),
]

# ---------------------------------------------------------------------------
# Entity keyword maps (saved alongside the model so the API uses same data)
# ---------------------------------------------------------------------------
SUBJECT_ENTITIES = {
    "math": ["math", "maths", "mathematics", "fraction", "fractions",
             "percentage", "percentages", "geometry", "algebra", "arithmetic",
             "number", "calculation", "equation"],
    "science": ["science", "biology", "physics", "chemistry",
                "photosynthesis", "gravity", "atom", "atoms", "molecule",
                "energy", "force", "cell", "cells", "ecosystem"],
    "english": ["english", "grammar", "essay", "comprehension",
                "vocabulary", "writing", "reading", "sentence", "verb",
                "noun", "tense"],
}

TOPIC_TIPS = {
    "fractions": {
        "tip": "Line up the denominators first — once they match, you only add or subtract the numerators. Think of the denominator as the size of each slice, and the numerator as how many slices you have.",
        "practice": [
            {"q": "What is 1/4 + 2/4?", "a": "3/4", "hint": "Same denominator — just add the numerators."},
            {"q": "Simplify 6/8.", "a": "3/4", "hint": "Divide both top and bottom by 2."},
            {"q": "What is 3/5 - 1/5?", "a": "2/5", "hint": "Same denominator — subtract numerators."},
        ]
    },
    "percentage": {
        "tip": "Think of 'percent' as 'per hundred'. So 20% = 20/100 = 0.20. To find 20% of 150, multiply: 0.20 × 150 = 30.",
        "practice": [
            {"q": "What is 10% of 200?", "a": "20", "hint": "10% = 0.10, so 0.10 × 200."},
            {"q": "What is 25% of 80?", "a": "20", "hint": "25% = 1/4, so divide 80 by 4."},
            {"q": "If a shirt costs ₹500 and is on 20% discount, how much do you save?", "a": "₹100", "hint": "20% of 500 = 0.20 × 500."},
        ]
    },
    "geometry": {
        "tip": "Always draw the shape before solving. Most geometry mistakes come from working blind. Label all given measurements on your sketch first.",
        "practice": [
            {"q": "What is the area of a rectangle 6 cm × 4 cm?", "a": "24 cm²", "hint": "Area = length × width."},
            {"q": "What is the perimeter of a square with side 5 cm?", "a": "20 cm", "hint": "Perimeter = 4 × side."},
            {"q": "A triangle has base 8 cm and height 5 cm. What is its area?", "a": "20 cm²", "hint": "Area = ½ × base × height."},
        ]
    },
    "gravity": {
        "tip": "Gravity is a pull between any two masses — the bigger the mass, the stronger the pull. Earth's gravity pulls everything toward its centre at about 9.8 m/s² (we round to 10 m/s²).",
        "practice": [
            {"q": "If you drop a ball and a feather in a vacuum, which hits the ground first?", "a": "Both at the same time", "hint": "Without air resistance, gravity acts equally on all masses."},
            {"q": "What is the weight of a 5 kg object on Earth? (g = 10 m/s²)", "a": "50 N", "hint": "Weight = mass × g."},
        ]
    },
    "photosynthesis": {
        "tip": "Plants take in CO₂ and water, and using sunlight, convert them into glucose (food) and oxygen. The chlorophyll in leaves captures the sunlight energy.",
        "practice": [
            {"q": "What gas do plants take in during photosynthesis?", "a": "Carbon dioxide (CO₂)", "hint": "Plants absorb CO₂ through tiny pores called stomata."},
            {"q": "What is the main product plants make during photosynthesis?", "a": "Glucose", "hint": "Glucose is the plant's food/energy store."},
            {"q": "Which part of the plant contains chlorophyll?", "a": "Leaves (chloroplasts)", "hint": "Chlorophyll is the green pigment that captures sunlight."},
        ]
    },
    "grammar": {
        "tip": "Read the sentence aloud — your ear catches grammar mistakes before your eyes do. Then check: does each verb agree with its subject? Are the tenses consistent?",
        "practice": [
            {"q": "Which is correct? 'She don't know' or 'She doesn't know'?", "a": "She doesn't know", "hint": "Third person singular uses 'doesn't', not 'don't'."},
            {"q": "Fill in the blank: 'He __ (go) to school yesterday.'", "a": "went", "hint": "'Yesterday' signals past tense — use the past form of 'go'."},
        ]
    },
    "algebra": {
        "tip": "Algebra is about finding the unknown. Whatever you do to one side of the equation, do to the other side too — that keeps the balance.",
        "practice": [
            {"q": "Solve: x + 5 = 12", "a": "x = 7", "hint": "Subtract 5 from both sides."},
            {"q": "Solve: 3x = 18", "a": "x = 6", "hint": "Divide both sides by 3."},
        ]
    },
    "atom": {
        "tip": "An atom has a nucleus (protons + neutrons) in the centre, with electrons orbiting around it. Protons carry a positive charge, electrons carry a negative charge, neutrons are neutral.",
        "practice": [
            {"q": "What particles are found in the nucleus of an atom?", "a": "Protons and neutrons", "hint": "Remember: the nucleus is in the centre."},
            {"q": "Which particle has a negative charge?", "a": "Electron", "hint": "Electrons orbit the nucleus."},
        ]
    },
}


def build_and_train():
    texts, labels = zip(*TRAINING_DATA)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    # Cross-validate to report generalisation accuracy
    scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")
    print(f"[Train] 5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(texts, labels)

    # Quick sanity checks on held-out phrasing the model hasn't seen
    test_cases = [
        ("yo what's up", "greeting"),
        ("i am totally lost on fractions", "ask_help"),
        ("give me a problem to solve", "request_practice"),
        ("that explanation was perfect thanks", "say_thanks"),
        ("i still don't understand", "still_confused"),
        ("see you tomorrow", "goodbye"),
    ]
    print("\n[Sanity checks]")
    for text, expected in test_cases:
        pred = pipeline.predict([text])[0]
        proba = pipeline.predict_proba([text]).max()
        status = "✓" if pred == expected else "✗"
        print(f"  {status} '{text}' → {pred} (conf {proba:.2f}), expected {expected}")

    return pipeline


def save_artifacts(pipeline, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "intent_classifier.pkl"), "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[Save] intent_classifier.pkl written to {out_dir}")

    entity_data = {
        "subjects": SUBJECT_ENTITIES,
        "topics": {k: v["tip"] for k, v in TOPIC_TIPS.items()},
    }
    with open(os.path.join(out_dir, "entity_data.json"), "w") as f:
        json.dump(entity_data, f, indent=2)

    with open(os.path.join(out_dir, "topic_tips.json"), "w") as f:
        json.dump(TOPIC_TIPS, f, indent=2)

    print("[Save] entity_data.json and topic_tips.json written")


if __name__ == "__main__":
    print("=== GyanMitra — Training Intent Classifier ===\n")
    model = build_and_train()
    save_artifacts(model, os.path.join(os.path.dirname(__file__), "artifacts"))
    print("\n✅ Training complete.")
