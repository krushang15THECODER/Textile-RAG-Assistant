import os
import re
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

from src.citations import extract_citations, is_abstention
from src.retrieval import format_context_for_prompt

load_dotenv(override=True)

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_INSTRUCTION = """You are an expert Textile Technical Consultant and Product Specialist for LabhTex Laces, assisting textile engineers, product designers, and garment manufacturers.
Your goal is to provide authoritative, helpful, and concise technical answers.

TONE & COMMUNICATION:
1. Speak naturally and directly as an industry expert.
2. NEVER use robotic meta-language or framing phrases such as:
   - "Based on the provided passages..."
   - "According to the context..."
   - "In the provided documents..."
   - "As mentioned in the text..."
   - "These are the applications mentioned in the passages..."
   Instead, answer the user's question directly with immediate clarity.
3. Organize information cleanly using bullet points, bold headings, and structured summaries.
4. Do NOT insert inline bracketed file citations (e.g. [document.pdf, Page 3]) inside sentences; all sources and page numbers are automatically extracted and shown in the UI.

GROUNDING & RELIABILITY:
1. Ensure all technical data, yarn specifications, lace constructions, and garment recommendations are completely faithful to the facts in the provided context.
2. If the context does not contain enough information to answer the question, state plainly:
   "The provided documents do not contain sufficient information to answer this question."
3. Do not invent ungrounded ASTM/ISO standards, chemical concentrations, or fictitious metrics.
"""


def resolve_llm_provider(api_key: Optional[str] = None) -> Dict[str, str]:
    """Detect active provider, API key, and model based on environment or passed key."""
    env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    # If explicit key was passed in (including empty string for testing)
    if api_key is not None:
        clean_key = api_key.strip()
        if clean_key.startswith("sk-or-"):
            return {
                "provider": "openrouter",
                "key": clean_key,
                "model": os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
            }
        else:
            return {
                "provider": "gemini",
                "key": clean_key,
                "model": os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            }

    # If provider is explicitly configured as openrouter
    if env_provider == "openrouter":
        return {
            "provider": "openrouter",
            "key": openrouter_key or gemini_key,
            "model": os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        }

    # If OPENROUTER_API_KEY is configured
    if openrouter_key and openrouter_key != "your_openrouter_api_key_here":
        return {
            "provider": "openrouter",
            "key": openrouter_key,
            "model": os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        }

    # Auto-detect OpenRouter key format if pasted into GEMINI_API_KEY by mistake
    if gemini_key and gemini_key.startswith("sk-or-"):
        return {
            "provider": "openrouter",
            "key": gemini_key,
            "model": os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        }

    # Default to Gemini
    return {
        "provider": "gemini",
        "key": gemini_key,
        "model": os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    }


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """Initialize official Google GenAI client from API key."""
    key = os.getenv("GEMINI_API_KEY") if api_key is None else api_key
    if not key or not key.strip() or key.strip() == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is missing or invalid. Please configure it or OPENROUTER_API_KEY in your .env file."
        )
    return genai.Client(api_key=key.strip())


def generate_with_openrouter(
    prompt: str,
    api_key: str,
    model_name: str
) -> str:
    """Call OpenRouter completions API endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Krushang15/Textile-RAG-Assistant",
        "X-Title": "Textile RAG Assistant",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=45)
    if resp.status_code != 200:
        error_detail = resp.text
        try:
            err_json = resp.json()
            if "error" in err_json and "message" in err_json["error"]:
                error_detail = err_json["error"]["message"]
        except Exception:
            pass
        raise RuntimeError(f"OpenRouter API Error ({resp.status_code}): {error_detail}")

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        return "No response generated by OpenRouter."
    return choices[0].get("message", {}).get("content", "").strip()


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Generate fact-grounded response using Gemini or OpenRouter from retrieved context passages."""
    active = resolve_llm_provider(api_key=api_key)
    provider = active["provider"]
    key = active["key"]
    raw_model = model_name or active["model"]

    if provider == "gemini":
        target_model = "gemini-3.8-flash" if "2.5-flash" in raw_model else raw_model
    else:
        target_model = raw_model

    if not retrieved_chunks:
        return {
            "answer": "The provided documents do not contain sufficient information to answer this question. No matching document passages were found in the knowledge base.",
            "citations": [],
            "is_grounded": False,
            "abstained": True,
            "model": target_model,
            "provider": provider,
            "error": None
        }

    formatted_context = format_context_for_prompt(retrieved_chunks)
    citations = extract_citations(retrieved_chunks)

    prompt = f"""TECHNICAL REFERENCE CONTEXT:
{formatted_context}

-----------------------------
USER INQUIRY:
{query}

EXPERT RESPONSE:"""

    try:
        if not key or not key.strip() or key in ("your_gemini_api_key_here", "your_openrouter_api_key_here"):
            raise ValueError(
                "GEMINI_API_KEY is missing or invalid. Please configure it or OPENROUTER_API_KEY in your .env file."
            )

        if provider == "openrouter":
            answer_text = generate_with_openrouter(prompt, api_key=key, model_name=target_model)
        else:
            import concurrent.futures
            client = get_gemini_client(api_key=key)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    client.models.generate_content,
                    model=target_model,
                    contents=prompt,
                    config={
                        "system_instruction": SYSTEM_INSTRUCTION,
                        "temperature": 0.1,
                        "max_output_tokens": 1024,
                    }
                )
                try:
                    response = future.result(timeout=30)
                    answer_text = response.text if response and response.text else "No response generated."
                except concurrent.futures.TimeoutError:
                    raise RuntimeError("Gemini API call timed out after 30 seconds. Please check API quota or switch to OpenRouter.")

        # Clean any accidental inline bracketed filenames (e.g., "[manual.pdf, Page 3]")
        answer_text = re.sub(r'\s*\[[^\]]*\.(?:pdf|docx|txt|xlsx)[^\]]*\]', '', answer_text).strip()

        abstained = is_abstention(answer_text)

        # Strip robotic meta-framing (e.g. "Based on the provided passages...") so the answer is direct
        if not abstained:
            answer_text = re.sub(
                r'^(?:Based on|According to|From|In|As per)\s+(?:the\s+)?(?:provided|given|retrieved|context|document|above)?\s*(?:passages?|text|context|documents?|information)?\s*[:,]?\s*(?:the\s+)?',
                '',
                answer_text,
                flags=re.IGNORECASE
            ).strip()
            if answer_text and answer_text[0].islower():
                answer_text = answer_text[0].upper() + answer_text[1:]

            answer_text = re.sub(
                r'\n+(?:These are|This is|The above are)\s+.*?(?:passages?|context|documents?)\.?\s*$',
                '',
                answer_text,
                flags=re.IGNORECASE
            ).strip()

        return {
            "answer": answer_text,
            "citations": citations if not abstained else [],
            "is_grounded": not abstained,
            "abstained": abstained,
            "model": target_model,
            "provider": provider,
            "error": None
        }

    except ValueError as val_err:
        return {
            "answer": f"Configuration Error: {val_err}",
            "citations": citations,
            "is_grounded": False,
            "abstained": False,
            "model": target_model,
            "provider": provider,
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
            "provider": provider,
            "error": err_msg
        }
    except Exception as exc:
        err_msg = f"LLM Generation Error: {exc}"
        return {
            "answer": err_msg,
            "citations": citations,
            "is_grounded": False,
            "abstained": False,
            "model": target_model,
            "provider": provider,
            "error": err_msg
        }
