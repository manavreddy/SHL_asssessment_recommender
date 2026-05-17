import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from ollama import Client


load_dotenv()

client = Client(
    host= "https://ollama.com",
    headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")},
)

MODEL = os.environ.get("OLLAMA_MODEL")


REFINE_EXTRACTION_PROMPT = """
You are an SHL assessment refinement planner.

The user is giving feedback on a previous assessment shortlist.
Extract the information needed to retrieve an updated shortlist.

Return ONLY valid JSON with this schema:
{{
  "keywords": ["positive retrieval keywords for the updated shortlist"],
  "previous_assessments": ["assessment names previously recommended by the assistant that are still relevant"]
}}

Rules:
- Use the previous assistant message to identify prior assessment names.
- Use the latest user message to identify what to add, remove, or change.
- Do not include terms that follow phrases like "do not use", "don't use", "drop", "remove", "exclude", "avoid", "without", or "no".
- If the user says "drop REST and add AWS", include AWS in keywords and do not include REST anywhere.
- Do not invent SHL assessment names.
- Do not explain your reasoning.

Previous assistant message:
{assistant_message}

Latest user feedback:
{user_message}
"""


def refine_agent(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    from tools.retriever import retrieve_assessments

    latest_feedback = _latest_user_message(conversation)
    previous_reply = _last_assistant_message(conversation)

    if not latest_feedback.strip():
        return {
            "action": "clarify",
            "reply": "What would you like me to change in the previous shortlist?",
            "items": [],
            "end_of_conversation": False,
        }

    extraction = _extract_refinement_inputs(previous_reply, latest_feedback)
    if extraction is None:
        print("Refine agent: LLM extraction failed, cannot refine without extracted JSON.")
        return {
            "action": "clarify",
            "reply": "I could not understand the refinement request. Please say what to add or remove from the shortlist.",
            "items": [],
            "end_of_conversation": False,
        }

    query = _build_query(extraction, latest_feedback)

    retrieved = retrieve_assessments(query, top_k=15)
    items = _normalize_items(retrieved)
    items = _dedupe_by_url(items)
    items = items[:5]

    if not items:
        return {
            "action": "clarify",
            "reply": "I could not find a better refined shortlist. Could you tell me what to add or remove?",
            "items": [],
            "keywords": extraction["keywords"],
            "previous_assessments": extraction["previous_assessments"],
            "end_of_conversation": False,
        }

    return {
        "action": "recommend",
        "reply": "",
        "items": items,
        "keywords": extraction["keywords"],
        "previous_assessments": extraction["previous_assessments"],
        "end_of_conversation": True,
    }


def _extract_refinement_inputs(
    previous_reply: str,
    latest_feedback: str,
) -> Dict[str, List[str]] | None:
    llm_result = _extract_with_llm(previous_reply, latest_feedback)
    if not llm_result:
        return None

    keywords = _as_list(llm_result.get("keywords"))
    previous_assessments = _as_list(llm_result.get("previous_assessments"))

    return {
        "keywords": _dedupe_strings(keywords),
        "previous_assessments": _dedupe_strings(previous_assessments),
    }


def _extract_with_llm(previous_reply: str, latest_feedback: str) -> Dict[str, Any]:
    if not MODEL:
        print("Refine agent: OLLAMA_MODEL is not set.")
        return {}

    prompt = REFINE_EXTRACTION_PROMPT.format(
        assistant_message=previous_reply,
        user_message=latest_feedback,
    )

    try:
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        content = response.message.content.strip()
        return json.loads(_json_object(content))
    except Exception as error:
        print(f"Refine extraction error: {error}")
        return {}


def _build_query(extraction: Dict[str, List[str]], latest_feedback: str) -> str:
    parts = []
    parts.extend(extraction["keywords"])
    parts.extend(extraction["previous_assessments"])

    if not parts:
        parts.append(latest_feedback)

    return " ".join(parts)


def _normalize_items(retrieved_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for item in retrieved_items:
        items.append(
            {
                "name": item.get("name", ""),
                "url": item.get("url", ""),
                "test_type": item.get("test_type", ""),
                "duration": item.get("duration", ""),
                "description": item.get("description", ""),
                "similarity_score": item.get("similarity_score"),
            }
        )
    return items


def _latest_user_message(conversation: List[Dict[str, str]]) -> str:
    for message in reversed(conversation):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _last_assistant_message(conversation: List[Dict[str, str]]) -> str:
    for message in reversed(conversation):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def _dedupe_by_url(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for item in items:
        url = item.get("url")
        if url in seen:
            continue
        seen.add(url)
        unique.append(item)
    return unique


def _dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value).strip()]


def _json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text
