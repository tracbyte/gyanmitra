"""
GyanMitra — Dialog Manager
Maps (intent, entities, context) → bot reply + updated context.
This mirrors the Dialog tab in IBM watsonx Assistant: each block below
is equivalent to a dialog node with conditions and a response.

The escalation threshold is set at confusion_count >= 2 for a topic.
"""

import random
from .nlu import TOPIC_TIPS

ESCALATION_THRESHOLD = 2


def _random(options):
    return random.choice(options)


def build_reply(intent: str, subject: str, topic: str, session: dict) -> tuple[str, bool]:
    """
    Returns (reply_text, escalated_flag).
    `session` is mutated in place: subject is updated, confusion counts incremented.
    """
    escalated = False

    # ── Update context (mirrors $context in watsonx) ──────────────────
    if subject:
        session["subject"] = subject

    effective_subject = subject or session.get("subject")

    if intent == "ask_help" and topic:
        confusion = session.setdefault("confusion", {})
        confusion[topic] = confusion.get(topic, 0) + 1

    if intent == "still_confused" and topic:
        confusion = session.setdefault("confusion", {})
        confusion[topic] = confusion.get(topic, 0) + 1

    # ── Check escalation ──────────────────────────────────────────────
    confusion_count = session.get("confusion", {}).get(topic, 0) if topic else 0
    if confusion_count >= ESCALATION_THRESHOLD:
        escalated = True

    # ── Dialog nodes ──────────────────────────────────────────────────

    if intent == "greeting":
        replies = [
            "Hi! I'm GyanMitra, your study support assistant. Which subject are you working on today?",
            "Hello! Great to see you. Tell me what you're studying and I'll help you out.",
            "Namaste! I'm GyanMitra. Ask me anything about your school subjects or request a practice question.",
        ]
        return _random(replies), escalated

    if intent == "ask_help":
        if topic:
            tip_data = TOPIC_TIPS.get(topic, {})
            tip = tip_data.get("tip", f"Let me look into {topic} with you.")
            if escalated:
                note = (
                    f"\n\n⚠️ You've asked about **{topic}** a few times now. "
                    "I've flagged this for your teacher so they can give you extra help."
                )
            else:
                note = ""
            return f"📚 **{topic.title()}**\n\n{tip}{note}", escalated

        if effective_subject:
            topics_for_subject = _topics_for_subject(effective_subject)
            return (
                f"Sure! Which {effective_subject} topic specifically? "
                f"For example: {', '.join(topics_for_subject[:3])}. "
                "Just type the topic name."
            ), escalated

        return (
            "Happy to help! Which subject is this about — Maths, Science, or English? "
            "Or just describe what you're stuck on."
        ), escalated

    if intent == "still_confused":
        if topic:
            tip_data = TOPIC_TIPS.get(topic, {})
            tip = tip_data.get("tip", "")
            practice = tip_data.get("practice", [])
            reply = f"No problem — let me try a different angle for **{topic.title()}**.\n\n{tip}"
            if practice:
                q = random.choice(practice)
                reply += f"\n\nTry this: {q['q']}\n💡 Hint: {q['hint']}"
            if escalated:
                reply += (
                    f"\n\n⚠️ Since {topic} keeps coming up, I've flagged it for your teacher too."
                )
            return reply, escalated

        return (
            "I understand — let me try explaining it differently. "
            "Can you tell me the exact topic or share the question you're stuck on?"
        ), escalated

    if intent == "request_practice":
        if not effective_subject:
            return "Which subject should the question be from — Maths, Science, or English?", escalated

        topics = _topics_for_subject(effective_subject)
        if not topics:
            return f"I don't have practice questions for {effective_subject} yet, but more are coming soon!", escalated

        chosen_topic = random.choice(topics)
        tip_data = TOPIC_TIPS.get(chosen_topic, {})
        practice = tip_data.get("practice", [])
        if practice:
            q = random.choice(practice)
            reply = (
                f"Here's a **{effective_subject.title()} — {chosen_topic.title()}** question:\n\n"
                f"❓ {q['q']}\n\n"
                f"Take your time. Type your answer when ready, or say 'hint' for a clue."
            )
        else:
            reply = (
                f"Here's a {chosen_topic} question: (full question bank coming in v2 — "
                "this slot would be filled by your school's question bank)."
            )
        return reply, escalated

    if intent == "say_thanks":
        replies = [
            "Anytime! Want to keep going or try a practice question?",
            "Glad that helped! Say 'practice' whenever you want to test yourself.",
            "You're welcome! Is there anything else you'd like to go through?",
        ]
        return _random(replies), escalated

    if intent == "goodbye":
        return (
            "Good session! Come back whenever you're stuck — I'll remember where we left off. "
            "Keep studying! 📖"
        ), escalated

    # fallback
    return (
        "I didn't quite catch that. You can:\n"
        "• Ask for help with a topic (e.g. 'explain fractions')\n"
        "• Request a practice question (e.g. 'quiz me on science')\n"
        "• Or just tell me what subject you're working on."
    ), escalated


def _topics_for_subject(subject: str) -> list[str]:
    """Return topics that belong to the given subject."""
    mapping = {
        "math": ["fractions", "percentage", "geometry", "algebra"],
        "science": ["gravity", "photosynthesis", "atom"],
        "english": ["grammar"],
    }
    return mapping.get(subject, list(TOPIC_TIPS.keys()))
