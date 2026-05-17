from typing import Any, Dict, List
from core.classify_intent import classify_intent
from core.response_generation import generate_final_response
from specialized_agents.clarification_agent import clarification_agent
from specialized_agents.comparison_agent import comparison_agent
from specialized_agents.refine_agent import refine_agent
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
    if intent == "refine":
        return refine_agent(conversation)
    return recommendation_agent(conversation)


def orchestrate(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    # ── 0. Guard: empty conversation ────────────────────────────────────────
    if not conversation:
        return {
            "reply": "Hello! Tell me the role you are hiring for and the key skills you need to assess.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # ── 1. Count user question turns ─────────────────────────────────────────
    turn_count = sum(
        1
        for msg in conversation
        if msg.get("role") == "user" and "?" in msg.get("content", "")
    )

    # ── 2. Route by turn-count override or classify intent ───────────────────
    
    intent = classify_intent(conversation)

    if intent == "refuse":
        return {
            "reply": _REFUSAL_REPLY,
            "recommendations": [],
            "end_of_conversation": False,
        }
    if turn_count == 8 and intent != "repeat":
        intent = "recommend"

    # ── 3. Route to sub-agent ────────────────────────────────────────────────
    agent_output = _route_to_agent(intent, conversation)
    # ── 3. Generate final reply ──────────────────────────────────────────────
    reply = generate_final_response(conversation, intent, agent_output)

    # ── 4. Convert items to recommendation dicts ─────────────────────────────
    def _norm_item(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return {
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "url": item.get("url", ""),
                "test_type": item.get("test_type", ""),
                "duration": item.get("duration", ""),
                "keys": item.get("keys", []),
                "languages": item.get("languages", []),
            }
        else:
            return {
                "name": getattr(item, "name", ""),
                "description": getattr(item, "description", ""),
                "url": getattr(item, "url", ""),
                "test_type": getattr(item, "test_type", ""),
                "duration": getattr(item, "duration", ""),
                "keys": getattr(item, "keys", []),
                "languages": getattr(item, "languages", []),
            }

    recommendations = [_norm_item(item) for item in agent_output.get("items", [])]

    has_recommendations = len(recommendations) > 0

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": True if (intent == "repeat" or turn_count == 8) else False,
    }
