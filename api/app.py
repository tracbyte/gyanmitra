"""
GyanMitra — Flask REST API
Endpoints:
  POST /api/chat              — main chat endpoint
  GET  /api/session/<sid>     — retrieve session state + history
  GET  /api/escalations       — teacher dashboard: open escalations
  POST /api/escalations/<id>/resolve  — mark an escalation resolved
  GET  /api/health            — liveness check
"""

import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

from . import db
from . import nlu as NLU
from .dialog import build_reply

app = Flask(__name__)
CORS(app)   # allow the standalone HTML frontend to call the API

db.init_db()


# ── /api/health ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "GyanMitra IVA API"})


# ── /api/chat ──────────────────────────────────────────────────────────────

@app.post("/api/chat")
def chat():
    """
    Request body (JSON):
      {
        "student_id": "student_abc",   // required; any unique string
        "session_id": "sess_xyz",      // required; generate client-side with uuid
        "message":    "help with fractions"
      }

    Response:
      {
        "reply":      "...",
        "intent":     "ask_help",
        "confidence": 0.82,
        "subject":    "math",
        "topic":      "fractions",
        "escalated":  false,
        "session":    { subject, turns, confusion }
      }
    """
    body = request.get_json(silent=True) or {}
    student_id = body.get("student_id", "").strip()
    session_id = body.get("session_id", "").strip()
    message = body.get("message", "").strip()

    if not student_id or not session_id:
        return jsonify({"error": "student_id and session_id are required"}), 400
    if not message:
        return jsonify({"error": "message is required"}), 400

    # Load or create session
    session = db.get_or_create_session(session_id, student_id)

    # NLU
    result = NLU.analyse(message)

    # Merge NLU subject with session subject (NLU wins if present)
    effective_subject = result.subject or session.get("subject")

    # Dialog
    reply, escalated = build_reply(
        intent=result.intent,
        subject=result.subject,
        topic=result.topic,
        session=session,
    )

    # Update turn count
    session["turns"] = session.get("turns", 0) + 1

    # Persist session
    db.save_session(session)

    # Log turn
    db.log_turn(
        session_id=session_id,
        turn_number=session["turns"],
        user_text=message,
        intent=result.intent,
        confidence=result.confidence,
        subject=result.subject,
        topic=result.topic,
        bot_reply=reply,
        escalated=escalated,
    )

    # Create escalation record if needed
    if escalated and result.topic:
        db.create_escalation(
            student_id=student_id,
            session_id=session_id,
            topic=result.topic,
            count=session.get("confusion", {}).get(result.topic, 0),
        )

    return jsonify({
        "reply": reply,
        "intent": result.intent,
        "confidence": round(result.confidence, 3),
        "subject": result.subject,
        "topic": result.topic,
        "escalated": escalated,
        "session": {
            "subject": session.get("subject"),
            "turns": session.get("turns"),
            "confusion": session.get("confusion", {}),
        },
    })


# ── /api/session ───────────────────────────────────────────────────────────

@app.get("/api/session/<session_id>")
def get_session(session_id):
    session = db.get_session(session_id)
    if session is None:
        return jsonify({"error": "session not found"}), 404
    history = db.get_turn_history(session_id)
    return jsonify({"session": session, "history": history})


@app.get("/api/student/<student_id>/sessions")
def get_student_sessions(student_id):
    sessions = db.get_student_sessions(student_id)
    return jsonify({"sessions": sessions})


# ── /api/escalations ───────────────────────────────────────────────────────

@app.get("/api/escalations")
def get_escalations():
    items = db.get_open_escalations()
    return jsonify({"escalations": items, "count": len(items)})


@app.post("/api/escalations/<int:esc_id>/resolve")
def resolve_escalation(esc_id):
    db.resolve_escalation(esc_id)
    return jsonify({"resolved": True, "id": esc_id})


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5050)
