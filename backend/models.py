from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field

class UserRegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=12, max_length=128)
    admin_key: Optional[str] = None

class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

class LoginResponse(BaseModel):
    user: UserResponse
    token: str
    expires_at: datetime

class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(default="New Technical Inquiry", max_length=200)

class ConversationUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)

class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0

class MessageCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    api_key: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None

class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    user_id: str
    role: str
    content: str
    citations: Optional[List[Dict[str, Any]]] = None
    citations_md: Optional[str] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    created_at: datetime

class DocumentItemResponse(BaseModel):
    document_id: str
    filename: str
    file_size: int
    upload_date: str
    chunk_count: int
    page_count: Any
    status: str
