from typing import Any, Dict, List
from core.classify_intent import classify_intent
from core.response_generation import generate_final_response
from specialized_agents.clarification_agent import clarification_agent
from specialized_agents.comparison_agent import comparison_agent
from specialized_agents.recommendation_agent import recommendation_agent


_REFUSAL_REPLY = (
    "I'm sorry, but your request doesn't appear to be related to SHL assessments. "
    "I can only help you discover, compare, and choose from the SHL product catalog. "
    "Please ask me about hiring assessments, test types, or role-based recommendations."
)


def _route_to_agent(intent: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    """Dispatch conversation to the correct sub-agent based on intent."""
    if intent == "clarify":
        return clarification_agent(conversation)
    if intent == "compare":
        return comparison_agent(conversation)
    # intent == "recommend"
    return recommendation_agent(conversation)


def orchestrate(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    # ── 0. Guard: empty conversation ────────────────────────────────────────
    if not conversation:
        return {
            "reply": "Hello! Tell me the role you are hiring for and the key skills you need to assess.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # ── 1. Classify intent ───────────────────────────────────────────────────
    intent = classify_intent(conversation)

    if(intent == "refuse"):
        return {
            "reply": _REFUSAL_REPLY,
            "recommendations": [],
            "end_of_conversation": False,
        }

    # ── 2. Route to sub-agent ────────────────────────────────────────────────
    agent_output = _route_to_agent(intent, conversation)

    # ── 3. Generate final reply ──────────────────────────────────────────────
    reply = generate_final_response(conversation, intent, agent_output)

    # ── 4. Convert items to recommendation dicts ─────────────────────────────
    recommendations = [
        {
            "name": item.name,
            "url": item.url,
            "test_type": item.test_type,
            "duration": item.duration,
        }
        for item in agent_output.get("items", [])
    ]

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": False if intent == "clarify" else True,
    }
