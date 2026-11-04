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
Your goal is to provide authoritative, helpful, and concise technical answers strictly based on the provided technical documentation.

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
   "I couldn't find sufficient information in the available documents to answer this question."
3. Do NOT invent company policies, pricing, minimum order quantities (MOQ), delivery timelines, product availability, or technical specifications if they are absent from the provided context.
4. When conversation history is provided, use it strictly to understand user references (such as which fabric, standard, or lace type is being referred to), but base your factual answer exclusively on facts in the TECHNICAL REFERENCE CONTEXT. Never treat previous conversational claims as verified documentation.
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
    model_name: str,
    timeout: int = 45,
    system_instruction: Optional[str] = None
) -> str:
    """Call OpenRouter completions API endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/krushang15THECODER/Textile-RAG-Assistant",
        "X-Title": "Textile RAG Assistant",
        "Content-Type": "application/json"
    }
    sys_content = system_instruction if system_instruction is not None else SYSTEM_INSTRUCTION
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=timeout)
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


def format_history_for_prompt(history: Optional[List[Dict[str, str]]]) -> str:
    """Format chronological conversation turns into a clean dialogue block."""
    if not history:
        return ""
    lines = []
    for turn in history:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def is_dependent_followup(query: str) -> bool:
    """Check if query is dependent on prior conversation context."""
    q_lower = query.lower().strip()
    words = q_lower.split()

    # Short queries with pronouns or questions without explicit subject
    pronouns = {"it", "its", "they", "them", "their", "this", "that", "these", "those"}
    if any(p in words for p in pronouns):
        return True

    # Dependent prefixes
    dependent_prefixes = (
        "what about", "how about", "and what", "what is its", "what are its",
        "tell me more", "how much", "what is the price", "what does it cost",
        "is it", "can it", "why?", "where?", "how so?", "explain more",
        "what else"
    )
    if any(q_lower.startswith(prefix) for prefix in dependent_prefixes):
        return True

    # Very short question (e.g., "Price?", "Washing instructions?", "Colors?")
    if len(words) <= 3 and not any(term in q_lower for term in ("astm", "iso", "raschel", "leavers", "guipure", "chantilly", "lace")):
        return True

    return False


def rewrite_query_for_retrieval(
    query: str,
    history: Optional[List[Dict[str, str]]],
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """Convert a context-dependent follow-up question into an independent standalone query for vector retrieval.
    
    If query is already standalone or no history exists, returns the original query unchanged.
    """
    clean_query = query.strip()
    if not clean_query or not history:
        return clean_query

    # If question is long and self-contained without referential pronouns, skip rewriting
    if not is_dependent_followup(clean_query):
        return clean_query

    formatted_history = format_history_for_prompt(history[-4:])
    if not formatted_history:
        return clean_query

    active = resolve_llm_provider(api_key=api_key)
    provider = active["provider"]
    key = active["key"]
    raw_model = model or active["model"]

    prompt = f"""Given the following conversation history between a user and an AI textile specialist, and a follow-up inquiry, rewrite the follow-up inquiry into a single standalone, self-contained search query suitable for technical document retrieval.

RULES:
1. Preserve the user's specific intent.
2. Replace ambiguous pronouns ("it", "its", "they", "that") with the specific lace, fabric, or topic discussed in the history.
3. Do NOT answer the question.
4. Output ONLY the rewritten standalone query text. Do not add quotes or explanations.

CONVERSATION HISTORY:
{formatted_history}

FOLLOW-UP INQUIRY:
{clean_query}

STANDALONE SEARCH QUERY:"""

    try:
        if not key or key in ("your_gemini_api_key_here", "your_openrouter_api_key_here"):
            raise ValueError("No active LLM API key available for query rewriting.")

        if provider == "openrouter":
            rewritten = generate_with_openrouter(
                prompt=prompt,
                api_key=key,
                model_name=raw_model,
                timeout=6,
                system_instruction="You are a query rewriting assistant for technical search."
            )
        else:
            client = get_gemini_client(api_key=key)
            target_model = "gemini-3.8-flash" if "2.5-flash" in raw_model else raw_model
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 60,
                }
            )
            rewritten = response.text if response and response.text else clean_query

        # Clean output
        clean_rewritten = re.sub(r'^(?:Standalone Search Query|Rewritten Query|Search Query)\s*:\s*', '', rewritten, flags=re.IGNORECASE).strip().strip('"\'')
        if clean_rewritten and len(clean_rewritten) > 3:
            return clean_rewritten
        return clean_query

    except Exception as e:
        print(f"[REWRITE FALLBACK] Could not rewrite query via LLM ({e}), applying heuristic fallback.")
        # Heuristic fallback: find the most recent user turn or subject keywords
        last_user_query = ""
        for turn in reversed(history):
            if turn.get("role") == "user":
                last_user_query = turn.get("content", "")
                break
        if last_user_query:
            # Combine core subject of previous query with follow-up
            return f"{last_user_query.strip()} {clean_query}"
        return clean_query


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Generate fact-grounded response using Gemini or OpenRouter from retrieved context passages and conversation history."""
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
            "answer": "I couldn't find sufficient information in the available documents to answer this question. The provided documents do not contain sufficient information.",
            "citations": [],
            "is_grounded": False,
            "abstained": True,
            "model": target_model,
            "provider": provider,
            "error": None
        }

    formatted_context = format_context_for_prompt(retrieved_chunks)
    citations = extract_citations(retrieved_chunks)

    # Optional conversational history section
    history_block = ""
    if conversation_history:
        formatted_history = format_history_for_prompt(conversation_history[-4:])
        if formatted_history:
            history_block = f"""RECENT CONVERSATION HISTORY:
{formatted_history}

-----------------------------
"""

    prompt = f"""{history_block}TECHNICAL REFERENCE CONTEXT:
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
