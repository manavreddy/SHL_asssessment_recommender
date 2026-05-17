import json
import os

from typing import Any, Dict, List
from ollama import Client
from tools.get_assessment import get_assessment

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY', '')}
)

MODEL = os.environ.get("OLLAMA_MODEL")

SYSTEM_PROMPT = """
You are an SHL assessment comparison extraction assistant.

Your task:
Identify the TWO SHL assessments the user wants to compare.

Rules:
- Extract only assessment names explicitly or implicitly referenced.
- Preserve abbreviations like OPQ, GSA, Verify, etc.
- Ignore non-assessment entities.
- If only one assessment is mentioned, return it alone.
- If none are identifiable, return an empty list.

Return STRICT JSON ONLY.

Schema:
{
  "assessments": ["assessment_1", "assessment_2"]
}

Examples:

User:
Compare OPQ and GSA

Output:
{
  "assessments": ["OPQ", "GSA"]
}

User:
What is the difference between Verify Interactive and OPQ32r?

Output:
{
  "assessments": ["Verify Interactive", "OPQ32r"]
}
"""


def comparison_agent(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    # Step 1: Extract assessment names from conversation via LLM
    extraction = client.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *conversation],
        stream=False,
    )

    try:
        parsed = json.loads(extraction.message.content.strip())
        assessment_names = parsed.get("assessments", [])
    except Exception:
        assessment_names = []

    # Step 2: Fetch full raw JSON for each assessment from catalog
    matched = []
    for name in assessment_names:
        data = get_assessment(name)
        if data:
            matched.append(data)

    # Step 3: Need exactly 2 to compare
    if len(matched) < 2:
        return {
            "action": "compare",
            "reply": "I couldn't find both assessments in the catalog. Could you clarify the exact names?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    return {
        "action": "compare",
        "assessments": extraction,
        "catalog_data" : matched
    }
