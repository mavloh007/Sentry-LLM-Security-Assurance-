"""
Supabase Client for SGBank Withdrawal Chatbot
Handles authentication, conversations, embeddings, and audit logging
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from uuid import uuid4
import json

from supabase import create_client, Client
from postgrest import exceptions as postgrest_exceptions
from dotenv import load_dotenv


load_dotenv()


class SupabaseDB:
    """Wrapper for Supabase database operations with pgvector support"""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        """
        Initialize Supabase client
        
        Args:
            supabase_url: Supabase project URL (defaults to env var)
            supabase_key: Supabase service role key (defaults to env var)
        """
        self.url = supabase_url or os.getenv("SUPABASE_URL")
        self.key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env or passed as arguments"
            )

        self.client: Client = create_client(self.url, self.key)

    # ==================== USER MANAGEMENT ====================

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by ID"""
        try:
            response = self.client.table("users").select("*").eq("id", user_id).single().execute()
            return response.data if response.data else None
        except postgrest_exceptions.APIError as e:
            print(f"Error fetching user {user_id}: {e}")
            return None

    def create_user(self, user_id: str, email: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new user profile"""
        try:
            user_data = {
                "id": user_id,
                "email": email,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            response = self.client.table("users").insert(user_data).execute()
            return response.data[0] if response.data else user_data
        except postgrest_exceptions.APIError as e:
            print(f"Error creating user: {e}")
            return user_data

    def update_user_metadata(self, user_id: str, metadata: Dict[str, Any]) -> Optional[Dict]:
        """Update user metadata"""
        try:
            response = (
                self.client.table("users")
                .update({"metadata": metadata, "updated_at": datetime.utcnow().isoformat()})
                .eq("id", user_id)
                .execute()
            )
            return response.data[0] if response.data else None
        except postgrest_exceptions.APIError as e:
            print(f"Error updating user metadata: {e}")
            return None

    # ==================== CONVERSATION MANAGEMENT ====================

    def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a new conversation session"""
        try:
            conversation_id = str(uuid4())
            conversation_data = {
                "id": conversation_id,
                "user_id": user_id,
                "title": title or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            response = self.client.table("conversations").insert(conversation_data).execute()
            return response.data[0] if response.data else conversation_data
        except postgrest_exceptions.APIError as e:
            print(f"Error creating conversation: {e}")
            raise

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID"""
        try:
            response = (
                self.client.table("conversations")
                .select("*")
                .eq("id", conversation_id)
                .single()
                .execute()
            )
            return response.data if response.data else None
        except postgrest_exceptions.APIError as e:
            print(f"Error fetching conversation: {e}")
            return None

    def list_user_conversations(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all conversations for a user"""
        try:
            response = (
                self.client.table("conversations")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )
            return response.data if response.data else []
        except postgrest_exceptions.APIError as e:
            print(f"Error listing conversations: {e}")
            return []

    # Backwards-compatible alias (app.py currently calls this)
    def get_user_conversations(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        return self.list_user_conversations(user_id=user_id, limit=limit, offset=offset)

    def update_conversation_metadata(self, conversation_id: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Merge/update conversation metadata.

        Best-effort: reads existing metadata and merges keys, then updates.
        """
        try:
            existing = self.get_conversation(conversation_id)
            merged = {}
            if isinstance(existing, dict) and isinstance(existing.get("metadata"), dict):
                merged.update(existing["metadata"])
            merged.update(metadata or {})

            response = (
                self.client.table("conversations")
                .update({"metadata": merged, "updated_at": datetime.utcnow().isoformat()})
                .eq("id", conversation_id)
                .execute()
            )
            return response.data[0] if response.data else None
        except postgrest_exceptions.APIError as e:
            print(f"Error updating conversation metadata: {e}")
            return None

    # ==================== MESSAGE MANAGEMENT ====================

    def add_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,  # "user" or "assistant"
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Add a message to conversation"""
        try:
            message_data = {
                "id": str(uuid4()),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
            }
            response = self.client.table("messages").insert(message_data).execute()
            return response.data[0] if response.data else message_data
        except postgrest_exceptions.APIError as e:
            print(f"Error adding message: {e}")
            raise

    def get_conversation_history(
        self, conversation_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation message history"""
        try:
            response = (
                self.client.table("messages")
                .select("*")
                .eq("conversation_id", conversation_id)
                .order("created_at", desc=False)
                .limit(limit)
                .execute()
            )
            return response.data if response.data else []
        except postgrest_exceptions.APIError as e:
            print(f"Error fetching conversation history: {e}")
            return []

    # ==================== ACCOUNT SNAPSHOT (SIMPLE) ====================

    def get_user_account_snapshot(self, user_id: str) -> Dict[str, Any]:
        """Return simple account fields for demo tools.

        Supports both schemas:
        - `users.metadata` JSON object containing these fields
        - top-level columns on the `users` table (e.g., balance, daily_limit)
        """

        user = self.get_user(user_id)

        metadata: Dict[str, Any] = {}
        if isinstance(user, dict) and isinstance(user.get("metadata"), dict):
            metadata = user["metadata"]

        def pick(key: str, default: Any = None) -> Any:
            if metadata.get(key) is not None:
                return metadata.get(key)
            if isinstance(user, dict) and user.get(key) is not None:
                return user.get(key)
            return default

        return {
            "user_id": user_id,
            "balance": pick("balance"),
            "daily_limit": pick("daily_limit")
        }

    # ==================== DOCUMENT & VECTOR MANAGEMENT ====================

    def add_document(
        self,
        content: str,
        embedding: List[float],
        source: str,
        doc_type: str = "policy",
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Add document with embedding to vector store"""
        try:
            doc_data = {
                "id": str(uuid4()),
                "content": content,
                "embedding": embedding,  # pgvector will handle this
                "source": source,
                "doc_type": doc_type,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
            }
            response = self.client.table("documents").insert(doc_data).execute()
            return response.data[0] if response.data else doc_data
        except postgrest_exceptions.APIError as e:
            print(f"Error adding document: {e}")
            raise

    def search_documents(
        self,
        embedding: List[float],
        limit: int = 5,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search documents using vector similarity via Supabase RPC (pgvector).

        Requires the ``search_documents`` RPC function to be set up in Supabase.
        Returns an empty list on failure so the caller can degrade gracefully.
        """
        try:
            response = self.client.rpc(
                "search_documents",
                {
                    "query_embedding": embedding,
                    "match_limit": limit,
                    "match_threshold": threshold,
                },
            ).execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    def get_all_documents(self, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all documents (useful for upsert during init)"""
        try:
            query = self.client.table("documents").select("*")
            if doc_type:
                query = query.eq("doc_type", doc_type)
            response = query.execute()
            return response.data if response.data else []
        except postgrest_exceptions.APIError as e:
            print(f"Error fetching documents: {e}")
            return []

    # ==================== UTILITY METHODS ====================

    def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            response = self.client.table("users").select("id").limit(1).execute()
            return True
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get aggregated stats for a user"""
        try:
            messages = self.client.table("messages").select("id").eq("user_id", user_id).execute()
            message_count = len(messages.data) if messages.data else 0

            conversations = (
                self.client.table("conversations").select("id").eq("user_id", user_id).execute()
            )
            conversation_count = len(conversations.data) if conversations.data else 0

            return {
                "user_id": user_id,
                "message_count": message_count,
                "conversation_count": conversation_count,
            }
        except Exception as e:
            print(f"Error getting user stats: {e}")
            return {}


# ==================== BATCH OPERATIONS ====================

class SupabaseVectorStore:
    """Wrapper specifically for vector operations with embedded documents"""

    def __init__(self, db: SupabaseDB):
        self.db = db

    def bulk_add_documents(
        self, documents: List[Tuple[str, List[float], str]], doc_type: str = "policy"
    ) -> List[Dict[str, Any]]:
        """
        Add multiple documents in batch
        
        Args:
            documents: List of (content, embedding, source) tuples
            doc_type: Type of document
        
        Returns:
            List of inserted document metadata
        """
        results = []
        for content, embedding, source in documents:
            try:
                result = self.db.add_document(
                    content=content,
                    embedding=embedding,
                    source=source,
                    doc_type=doc_type,
                )
                results.append(result)
            except Exception as e:
                print(f"Error adding document from {source}: {e}")
                continue
        return results

    def search(
        self,
        embedding: List[float],
        limit: int = 5,
        threshold: float = 0.7,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search documents with optional filtering"""
        results = self.db.search_documents(embedding, limit, threshold)
        
        # Filter by doc_type if specified
        if doc_type:
            results = [r for r in results if r.get("doc_type") == doc_type]
        
        return results[:limit]
