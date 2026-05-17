import json
import os

from ollama import Client
from typing import Any, Dict, List
from tools.retriever import retrieve_assessments

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY', '')}
)

MODEL = os.environ.get("OLLAMA_MODEL")

KEYWORD_EXTRACTION_PROMPT = """
You are an SHL assessment retrieval planner.

Your task is to extract semantic retrieval concepts from a hiring conversation
to help retrieve the most relevant SHL assessments.

Focus on:
- role seniority
- leadership level
- technical skills
- behavioral competencies
- personality requirements
- cognitive requirements
- hiring purpose
- benchmarking
- development vs selection
- influencing style
- strategic thinking
- stakeholder interaction

VERY IMPORTANT:
- Expand implied concepts when appropriate.
- Infer assessment intent from context.
- Prefer semantic concepts over exact wording.
- Include personality and behavioral assessment concepts whenever leadership evaluation is discussed.
- Include benchmark and selection concepts when candidate comparison is mentioned.

Return ONLY a comma-separated keyword list.

GOOD OUTPUT:
executive leadership, personality assessment, behavioral assessment, strategic thinking, influencing style, leadership benchmark, executive selection

BAD OUTPUT:
The user needs leadership assessments.

Conversation:
{conversation_text}

Keywords:
"""


def _extract_all_user_text(conversation: List[Dict[str, str]]) -> str:
    """Extract all user messages from conversation."""
    user_messages = [msg["content"] for msg in conversation if msg.get("role") == "user"]
    return " ".join(user_messages)


def _extract_keywords_with_llm(conversation_text: str) -> str:
    """Use LLM to extract keywords from conversation."""
    try:
        prompt = KEYWORD_EXTRACTION_PROMPT.format(conversation_text=conversation_text)
        
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        
        keywords = response.message.content.strip()
        return keywords
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return conversation_text


def recommendation_agent(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    # Step 1: Extract keywords from conversation
    all_user_text = _extract_all_user_text(conversation)
    
    if not all_user_text.strip():
        return {
            "action": "clarify",
            "reply": "I could not find enough information. Which role, skills, and seniority should I focus on?",
            "items": [],
            "keywords": "",
            "end_of_conversation": False,
        }
    
    keywords = _extract_keywords_with_llm(all_user_text)
    
    # Step 2: Retrieve assessments using the retriever module
    query = keywords if keywords.strip() else all_user_text
    retrieved_items = retrieve_assessments(query, top_k=5)

    # Step 3: Convert retrieved items to plain dicts
    items: List[Dict[str, Any]] = []
    for item_dict in retrieved_items:
        try:
            item = {
                "name": item_dict.get("name", ""),
                "url": item_dict.get("url", ""),
                "test_type": item_dict.get("test_type", ""),
                "duration": item_dict.get("duration", ""),
                "description": item_dict.get("description", ""),
                "keys": item_dict.get("keys", []),
                "languages": item_dict.get("languages", []),
            }
            items.append(item)
        except Exception as e:
            print(f"Error creating item dict: {e}")
    
    # Step 4: Handle no results
    if not items:
        return {
            "action": "clarify",
            "reply": "I could not find strong matches yet. Could you provide more details about the role, skills, or specific assessment types?",
            "items": [],
            "keywords": keywords,
            "end_of_conversation": False,
        }

    return {
        "action": "recommend",
        "reply": "",
        "items": items,
        "keywords": keywords,
        "end_of_conversation": True,
    }