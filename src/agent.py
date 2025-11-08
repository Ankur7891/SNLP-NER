import yaml
import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.entity import extract_entities
from src.relation import extract_relations

load_dotenv()


def load_system_prompt() -> str:
    """Load system prompt text from YAML."""
    with open("prompts/system.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["system_prompt"]


def build_llm() -> ChatGoogleGenerativeAI:
    """Initialize Gemini model."""
    return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_GEMINI_MODEL"), temperature=0)


def clean_json_output(text: str) -> dict:
    """Remove markdown code fences and parse JSON safely."""
    if not text:
        return {}
    
    cleaned = re.sub(
        r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE
    ).strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_text": cleaned}


def run_agent(text: str) -> dict:
    """Run full pipeline: entity extraction + relation extraction."""
    llm = build_llm()
    system_prompt = load_system_prompt()

    entities_raw = extract_entities(llm, text, system_prompt)
    relations_raw = extract_relations(llm, text, system_prompt)

    entities_json = clean_json_output(entities_raw)
    relations_json = clean_json_output(relations_raw)

    return {
        "entities": entities_json.get("entities", {}),
        "relations": relations_json.get("relations", []),
    }
