from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from bson import ObjectId

from backend.database import get_db
from backend.models import ConversationCreateRequest, ConversationResponse, ConversationUpdateRequest
from backend.security import get_current_user

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


def parse_object_id(oid_str: str) -> ObjectId:
    try:
        return ObjectId(oid_str)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation ID format."
        )


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(req: ConversationCreateRequest, current_user: dict = Depends(get_current_user)):
    """Create a new conversation session for the authenticated user."""
    db = get_db()
    now = datetime.now(timezone.utc)
    user_id_str = str(current_user["_id"])

    conv_doc = {
        "user_id": user_id_str,
        "title": req.title or "New Technical Inquiry",
        "created_at": now,
        "updated_at": now
    }
    result = db.conversations.insert_one(conv_doc)
    conv_doc["_id"] = result.inserted_id

    return ConversationResponse(
        id=str(conv_doc["_id"]),
        user_id=user_id_str,
        title=conv_doc["title"],
        created_at=conv_doc["created_at"],
        updated_at=conv_doc["updated_at"],
        message_count=0
    )


@router.get("", response_model=List[ConversationResponse])
async def list_conversations(current_user: dict = Depends(get_current_user)):
    """List all conversations owned by the authenticated user, ordered by most recently updated."""
    db = get_db()
    user_id_str = str(current_user["_id"])

    cursor = db.conversations.find({"user_id": user_id_str}).sort("updated_at", -1)
    results = []

    for doc in cursor:
        conv_id_str = str(doc["_id"])
        msg_count = db.messages.count_documents({"conversation_id": conv_id_str})
        results.append(
            ConversationResponse(
                id=conv_id_str,
                user_id=doc["user_id"],
                title=doc.get("title", "Untitled Conversation"),
                created_at=doc.get("created_at", datetime.now(timezone.utc)),
                updated_at=doc.get("updated_at", datetime.now(timezone.utc)),
                message_count=msg_count
            )
        )
    return results


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Retrieve details for a specific conversation with strict ownership check."""
    db = get_db()
    oid = parse_object_id(conversation_id)
    user_id_str = str(current_user["_id"])

    conv = db.conversations.find_one({"_id": oid, "user_id": user_id_str})
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied."
        )

    msg_count = db.messages.count_documents({"conversation_id": str(oid)})
    return ConversationResponse(
        id=str(conv["_id"]),
        user_id=conv["user_id"],
        title=conv.get("title", "Untitled Conversation"),
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        message_count=msg_count
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def rename_conversation(
    conversation_id: str,
    req: ConversationUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Rename a conversation owned by the authenticated user."""
    db = get_db()
    oid = parse_object_id(conversation_id)
    user_id_str = str(current_user["_id"])

    conv = db.conversations.find_one({"_id": oid, "user_id": user_id_str})
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied."
        )

    now = datetime.now(timezone.utc)
    new_title = req.title.strip()

    db.conversations.update_one(
        {"_id": oid},
        {"$set": {"title": new_title, "updated_at": now}}
    )

    conv["title"] = new_title
    conv["updated_at"] = now
    msg_count = db.messages.count_documents({"conversation_id": str(oid)})

    return ConversationResponse(
        id=str(conv["_id"]),
        user_id=conv["user_id"],
        title=new_title,
        created_at=conv["created_at"],
        updated_at=now,
        message_count=msg_count
    )


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a conversation and purge its entire message history with ownership enforcement."""
    db = get_db()
    oid = parse_object_id(conversation_id)
    user_id_str = str(current_user["_id"])

    conv = db.conversations.find_one({"_id": oid, "user_id": user_id_str})
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied."
        )

    # Delete conversation and all its messages
    db.conversations.delete_one({"_id": oid})
    db.messages.delete_many({"conversation_id": str(oid)})

    return {"message": "Conversation and its message history deleted successfully."}
