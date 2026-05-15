import json
import os

from ollama import Client
from typing import Any, Dict, List
from models import CatalogItem
from tools.retriever import retrieve_assessments

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY', '')}
)

MODEL = os.environ.get("OLLAMA_MODEL")

KEYWORD_EXTRACTION_PROMPT = """Extract the most important keywords and phrases from this conversation that would help find relevant SHL assessments.

Focus on:
- Job roles/titles
- Technical skills
- Competencies and behaviors
- Seniority levels
- Industry context
- Assessment types (cognitive, personality, technical, etc.)

Return ONLY a comma-separated list of keywords, no explanations or extra text.

Conversation:
{conversation_text}

Keywords:"""


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
    """
    Recommendation agent: extract keywords and retrieve top 10 assessments via RAG.
    
    Parameters
    ----------
    conversation : list of {"role": str, "content": str}
        Full chat history.
    
    Returns
    -------
    dict with keys: action, reply, items, end_of_conversation, keywords (for RAG)
    """
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
    retrieved_items = retrieve_assessments(query, top_k=10)

    # Step 3: Convert retrieved items to CatalogItem objects
    items = []
    for item_dict in retrieved_items:
        try:
            item = CatalogItem(
                name=item_dict.get("name", ""),
                url=item_dict.get("url", ""),
                test_type=item_dict.get("test_type", ""),
                duration=item_dict.get("duration", ""),
                description=item_dict.get("description", ""),
            )
            items.append(item)
        except Exception as e:
            print(f"Error creating CatalogItem: {e}")
    
    # Step 4: Handle no results
    if not items:
        return {
            "action": "clarify",
            "reply": "I could not find strong matches yet. Could you provide more details about the role, skills, or specific assessment types?",
            "items": [],
            "keywords": keywords,
            "end_of_conversation": False,
        }
    
    # Step 5: Generate introduction message
    catalog_context = "\n".join(
        f"- {item.name} | {item.test_type} | {item.duration or 'duration not listed'} | {item.url}"
        for item in items
    )
    
    prompt = (
        "Write a short reply introducing this SHL assessment shortlist based on the conversation.\n"
        "Do not invent assessments, URLs, durations, or test types.\n"
        "Do not format the recommendations as JSON; the API will attach them separately.\n\n"
        f"User context:\n{all_user_text}\n\n"
        f"Keywords extracted:\n{keywords}\n\n"
        f"Catalog-backed shortlist:\n{catalog_context}"
    )
    
    try:
        reply = client.chat(
            model=MODEL,
            messages=[{
                "role": "system",
                "content": "You are a concise SHL assessment recommender grounded only in supplied catalog items."
            }, {
                "role": "user",
                "content": prompt
            }],
            stream=False,
        )
        reply_text = reply.message.content.strip()
    except Exception as e:
        print(f"Error generating reply: {e}")
        reply_text = f"Here is a catalog-backed SHL shortlist of {len(items)} assessment(s) that best match your requirements."
    
    return {
        "action": "recommend",
        "reply": reply_text or f"Here is a catalog-backed SHL shortlist of {len(items)} assessment(s).",
        "items": items,
        "keywords": keywords,
        "end_of_conversation": True
    }
