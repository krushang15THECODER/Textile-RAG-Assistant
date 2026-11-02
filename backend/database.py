from typing import Optional
import pymongo
from pymongo.database import Database
from backend.config import get_mongodb_uri, get_mongodb_db_name

class DatabaseUnavailableError(Exception):
    """Raised when MongoDB connection is missing or cannot be reached."""
    pass

_mongo_client: Optional[pymongo.MongoClient] = None
_db_override: Optional[Database] = None

def get_client() -> pymongo.MongoClient:
    """Return active MongoClient instance, or raise DatabaseUnavailableError."""
    global _mongo_client
    if _mongo_client is not None:
        try:
            _mongo_client.admin.command("ping")
            return _mongo_client
        except Exception:
            _mongo_client = None

    uri = get_mongodb_uri()
    if not uri:
        raise DatabaseUnavailableError(
            "MongoDB is unavailable: MONGODB_URI is not set in environment variables. "
            "Please configure MONGODB_URI to connect to MongoDB Atlas."
        )

    try:
        _mongo_client = pymongo.MongoClient(
            uri,
            serverSelectionTimeoutMS=4000,
            connectTimeoutMS=4000
        )
        # Test connection ping
        _mongo_client.admin.command("ping")
        return _mongo_client
    except Exception as e:
        _mongo_client = None
        raise DatabaseUnavailableError(f"MongoDB connection failed: {e}") from e

def get_db() -> Database:
    """Return MongoDB Database instance with required collections and indexes."""
    global _db_override
    if _db_override is not None:
        return _db_override

    client = get_client()
    db_name = get_mongodb_db_name()
    db = client[db_name]
    return db

def set_test_db(db_instance: Optional[Database]) -> None:
    """Set mock database instance for automated testing."""
    global _db_override
    _db_override = db_instance

def init_indexes(db: Database) -> None:
    """Ensure all required collections and performance/integrity indexes exist."""
    try:
        # Users collection
        db.users.create_index([("email", pymongo.ASCENDING)], unique=True)

        # Sessions collection
        db.sessions.create_index([("token_hash", pymongo.ASCENDING)], unique=True)
        db.sessions.create_index([("user_id", pymongo.ASCENDING)])
        db.sessions.create_index([("expires_at", pymongo.ASCENDING)])

        # Conversations collection
        db.conversations.create_index([("user_id", pymongo.ASCENDING), ("updated_at", pymongo.DESCENDING)])

        # Messages collection
        db.messages.create_index([("conversation_id", pymongo.ASCENDING), ("created_at", pymongo.ASCENDING)])
    except Exception as e:
        print(f"Warning during index initialization: {e}")
