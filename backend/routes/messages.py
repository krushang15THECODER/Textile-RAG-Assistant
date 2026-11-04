from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from bson import ObjectId

from backend.config import get_retrieval_score_threshold
from backend.database import get_db
from backend.models import MessageCreateRequest, MessageResponse
from backend.security import get_current_user

router = APIRouter(prefix="/api/conversations", tags=["Messages"])


def parse_object_id(oid_str: str) -> ObjectId:
    try:
        return ObjectId(oid_str)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation ID format."
        )


def generate_auto_title(prompt: str) -> str:
    """Generate a clean 5-8 word title from initial user prompt."""
    words = prompt.strip().split()
    if len(words) <= 6:
        title = " ".join(words)
    else:
        title = " ".join(words[:6]) + "..."
    return title[:60] if title else "Technical Inquiry"


@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve all messages for a conversation in chronological order with strict ownership check."""
    db = get_db()
    oid = parse_object_id(conversation_id)
    user_id_str = str(current_user["_id"])

    # Enforce conversation ownership
    conv = db.conversations.find_one({"_id": oid, "user_id": user_id_str})
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied."
        )

    cursor = db.messages.find({"conversation_id": str(oid)}).sort("created_at", 1)
    results = []

    for msg in cursor:
        results.append(
            MessageResponse(
                id=str(msg["_id"]),
                conversation_id=msg["conversation_id"],
                user_id=msg["user_id"],
                role=msg["role"],
                content=msg["content"],
                citations=msg.get("citations"),
                citations_md=msg.get("citations_md"),
                evidence=msg.get("evidence"),
                created_at=msg.get("created_at", datetime.now(timezone.utc))
            )
        )
    return results


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def post_message_and_generate_response(
    conversation_id: str,
    req: MessageCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Save user message, run RAG pipeline with follow-up awareness & thresholding, and persist response."""
    db = get_db()
    oid = parse_object_id(conversation_id)
    user_id_str = str(current_user["_id"])

    # Enforce conversation ownership
    conv = db.conversations.find_one({"_id": oid, "user_id": user_id_str})
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied."
        )

    now = datetime.now(timezone.utc)
    conv_id_str = str(oid)

    # 1. Retrieve recent chronological conversation history prior to this message (up to last 4 messages)
    history_cursor = db.messages.find(
        {"conversation_id": conv_id_str}
    ).sort("created_at", 1)
    raw_history = list(history_cursor)
    recent_history = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in raw_history[-4:]
    ]

    # 2. Save user message
    user_msg_doc = {
        "conversation_id": conv_id_str,
        "user_id": user_id_str,
        "role": "user",
        "content": req.content.strip(),
        "created_at": now
    }
    insert_res = db.messages.insert_one(user_msg_doc)
    user_msg_doc["_id"] = insert_res.inserted_id

    # 3. Check if conversation needs auto-titling (first message)
    existing_count = db.messages.count_documents({"conversation_id": conv_id_str, "role": "user"})
    if existing_count == 1 and (conv.get("title") in ("New Technical Inquiry", "Untitled Conversation", None)):
        auto_title = generate_auto_title(req.content)
        db.conversations.update_one(
            {"_id": oid},
            {"$set": {"title": auto_title, "updated_at": now}}
        )
    else:
        db.conversations.update_one(
            {"_id": oid},
            {"$set": {"updated_at": now}}
        )

    # 4. Execute RAG Pipeline with Query Rewriting and Score Thresholding
    try:
        from src.citations import format_citations_markdown
        from src.generation import generate_answer, rewrite_query_for_retrieval
        from src.retrieval import retrieve_context

        # Disambiguate follow-up queries if prior dialogue history exists
        search_query = req.content
        if recent_history:
            search_query = rewrite_query_for_retrieval(
                query=req.content,
                history=recent_history,
                api_key=req.api_key,
                model=req.model
            )
            if search_query != req.content:
                print(f"[RAG] Follow-up query rewritten: '{req.content}' -> '{search_query}'")

        threshold = get_retrieval_score_threshold()
        print(f"[RAG] Executing retrieval (threshold={threshold:.2f}) for: {search_query[:60]}...")
        retrieved_chunks = retrieve_context(
            query=search_query,
            top_k=4,
            score_threshold=threshold
        )

        # Deterministic Abstention Guardrail: Do NOT invoke LLM if no evidence meets threshold
        if not retrieved_chunks:
            print(f"[RAG] No evidence met similarity threshold ({threshold:.2f}). Returning clean abstention.")
            answer_text = "I couldn't find sufficient information in the available documents to answer this question."
            citations = []
            citations_md = ""
            retrieved_chunks = []
        else:
            print(f"[RAG] Retrieval passed with {len(retrieved_chunks)} chunks. Generating answer...")
            rag_result = generate_answer(
                query=req.content,
                retrieved_chunks=retrieved_chunks,
                conversation_history=recent_history,
                model_name=req.model,
                api_key=req.api_key
            )
            print("[RAG] Answer generation finished.")
            answer_text = rag_result.get("answer", "No response generated.")
            citations = rag_result.get("citations", [])
            citations_md = format_citations_markdown(citations) if citations else ""

    except Exception as rag_err:
        print(f"[RAG ERROR] Pipeline exception: {rag_err}")
        answer_text = f"An error occurred while generating the technical response: {rag_err}"
        citations = []
        citations_md = ""
        retrieved_chunks = []

    # 5. Save assistant response into MongoDB
    assistant_now = datetime.now(timezone.utc)
    assistant_msg_doc = {
        "conversation_id": conv_id_str,
        "user_id": user_id_str,
        "role": "assistant",
        "content": answer_text,
        "citations": citations,
        "citations_md": citations_md,
        "evidence": retrieved_chunks,
        "created_at": assistant_now
    }
    result = db.messages.insert_one(assistant_msg_doc)
    assistant_msg_doc["_id"] = result.inserted_id

    # 6. Update conversation updated_at timestamp
    db.conversations.update_one(
        {"_id": oid},
        {"$set": {"updated_at": assistant_now}}
    )

    return MessageResponse(
        id=str(assistant_msg_doc["_id"]),
        conversation_id=conv_id_str,
        user_id=user_id_str,
        role="assistant",
        content=answer_text,
        citations=citations,
        citations_md=citations_md,
        evidence=retrieved_chunks,
        created_at=assistant_now
    )
