# GyanMitra — Adaptive AI Learning Agent
**IBM SkillsBuild · Masterclass 5 · SDG 4: Quality Education**

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (index.html)                  │
│   Student Chat UI  ←→  Teacher Dashboard               │
└───────────────────────┬────────────────────────────────┘
                        │ REST API (JSON)
┌───────────────────────▼────────────────────────────────┐
│              Flask API  (api/app.py)                    │
│                                                         │
│  POST /api/chat          NLU → Dialog → Action          │
│  GET  /api/escalations   Teacher dashboard feed         │
│  POST /api/escalations/<id>/resolve                     │
│  GET  /api/session/<id>  Session + turn history         │
└────────────┬──────────────────────┬────────────────────┘
             │                      │
┌────────────▼──────────┐  ┌───────▼──────────────────┐
│   NLU Engine          │  │   SQLite Database         │
│   (api/nlu.py)        │  │   (data/gyanmitra.db)     │
│                       │  │                           │
│  TF-IDF Vectoriser    │  │  students                 │
│  Logistic Regression  │  │  sessions  (context)      │
│  Entity extraction    │  │  escalations              │
│                       │  │  turn_log  (audit trail)  │
└────────────┬──────────┘  └──────────────────────────┘
             │
┌────────────▼──────────┐
│   Dialog Manager      │
│   (api/dialog.py)     │
│                       │
│  Greeting node        │
│  Ask-help node        │
│  Still-confused node  │
│  Practice node        │
│  Escalation node      │
└───────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the intent classifier (creates models/artifacts/)
python models/train_intent_classifier.py

# 3. Start the Flask API
python run.py

# 4. Open frontend/index.html in your browser
#    (The page auto-detects the API and connects)
```

## What each file does

| File | Role |
|------|------|
| `models/train_intent_classifier.py` | Trains TF-IDF + Logistic Regression on 80+ labelled examples; saves `intent_classifier.pkl`, `entity_data.json`, `topic_tips.json` |
| `api/nlu.py` | Loads trained model; `analyse(text)` returns intent, confidence, subject, topic |
| `api/dialog.py` | Maps (intent, entities, context) → reply + escalation flag |
| `api/db.py` | SQLite helpers — students, sessions, escalations, turn_log tables |
| `api/app.py` | Flask REST API with 5 endpoints |
| `run.py` | Entry point; prints URL and starts server |
| `frontend/index.html` | Student chat UI + Teacher dashboard; calls API; graceful local fallback |

## API Reference

### POST /api/chat
```json
Request:  { "student_id": "s1", "session_id": "sess1", "message": "explain fractions" }
Response: { "reply": "...", "intent": "ask_help", "confidence": 0.80,
            "subject": "math", "topic": "fractions", "escalated": false,
            "session": { "subject": "math", "turns": 3, "confusion": {"fractions": 1} } }
```

### GET /api/escalations
Returns all open escalation records — topics where a student has been confused ≥ 2 times.

### POST /api/escalations/{id}/resolve
Mark an escalation as resolved (teacher has followed up).

## Topics Covered (v1)
**Maths:** fractions, percentage, geometry, algebra  
**Science:** photosynthesis, gravity, atom  
**English:** grammar

## Production Upgrade Path
1. Move NLU to IBM watsonx Assistant (same intent/entity design, real NLU model)
2. Add authentication (student login)
3. Multi-device sync via proper backend deployment
4. Expand question bank with regional teachers
5. Teacher dashboard as a separate React app
