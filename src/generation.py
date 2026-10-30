import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

from src.citations import extract_citations, is_abstention
from src.retrieval import format_context_for_prompt

load_dotenv(override=True)

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

SYSTEM_INSTRUCTION = """You are an expert Textile Technical Consultant assisting textile engineers and quality analysts.
Your task is to answer the user's question accurately and objectively using ONLY the provided document passages.

STRICT GROUNDING RULES:
1. Base your answer ENTIRELY on the provided context passages below.
2. If the context does NOT contain enough information to answer the question with certainty, state clearly:
   "The provided documents do not contain sufficient information to answer this question."
3. Do NOT extrapolate, speculate, or fabricate textile metrics, ASTM/ISO standards, chemical concentrations, or machine settings.
4. When stating facts, explicitly reference the source and page number in your narrative (e.g., "[weaving_specs.pdf, Page 2]").
5. Keep answers concise, factual, and professional.
"""


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """Initialize official Google GenAI client from API key."""
    key = os.getenv("GEMINI_API_KEY") if api_key is None else api_key
    if not key or not key.strip() or key.strip() == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is missing or invalid. Please configure it in your .env file."
        )
    return genai.Client(api_key=key.strip())


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Generate fact-grounded response using Gemini from retrieved context passages."""
    raw_model = model_name or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    target_model = "gemini-3.8-flash" if "2.5-flash" in raw_model else raw_model

    if not retrieved_chunks:
        return {
            "answer": "The provided documents do not contain sufficient information to answer this question. No matching document passages were found in the knowledge base.",
            "citations": [],
            "is_grounded": False,
            "abstained": True,
            "model": target_model,
            "error": None
        }

    formatted_context = format_context_for_prompt(retrieved_chunks)
    citations = extract_citations(retrieved_chunks)

    prompt = f"""CONTEXT PASSAGES:
{formatted_context}

-----------------------------
USER QUESTION:
{query}

ANSWER (grounded strictly in the context above):"""

    try:
        client = get_gemini_client(api_key=api_key)
        response = client.models.generate_content(
            model=target_model,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "temperature": 0.1,
                "max_output_tokens": 1024,
            }
        )

        answer_text = response.text if response and response.text else "No response generated."
        abstained = is_abstention(answer_text)

        return {
            "answer": answer_text,
            "citations": citations if not abstained else [],
            "is_grounded": not abstained,
            "abstained": abstained,
            "model": target_model,
            "error": None
        }

    except ValueError as val_err:
        return {
            "answer": f"Configuration Error: {val_err}",
            "citations": citations,
            "is_grounded": False,
            "abstained": False,
            "model": target_model,
            "error": str(val_err)
        }
    except APIError as api_err:
        err_msg = f"Gemini API Error ({api_err.code}): {api_err.message}"
        return {
            "answer": err_msg,
            "citations": citations,
            "is_grounded": False,
            "abstained": False,
            "model": target_model,
            "error": err_msg
        }
    except Exception as exc:
        err_msg = f"Unexpected generation error: {exc}"
        return {
            "answer": err_msg,
            "citations": citations,
            "is_grounded": False,
            "abstained": False,
            "model": target_model,
            "error": err_msg
        }
