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
        anon_key: Optional[str] = None,
    ):
        """
        Initialize Supabase client

        Args:
            supabase_url: Supabase project URL (defaults to env var)
            supabase_key: Supabase service role key (defaults to env var)
            anon_key: Supabase anon/public key used ONLY for transient auth
                clients during login/signup (defaults to env var).
        """
        self.url = supabase_url or os.getenv("SUPABASE_URL")
        self.key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY")
        # Anon key is optional at construction time so CLI/ingest scripts that
        # never call login/signup don't need it, but auth helpers below will
        # raise if it's missing when they're actually used.
        self.anon_key = anon_key or os.getenv("SUPABASE_ANON_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env or passed as arguments"
            )

        # The shared client is authenticated with the SERVICE ROLE key and its
        # auth header is never mutated after this point. All table reads/writes
        # go through it and therefore bypass RLS as trusted backend traffic.
        self.client: Client = create_client(self.url, self.key)

    # ==================== AUTH (TRANSIENT CLIENTS) ====================

    def _build_anon_client(self) -> Client:
        """Build a throwaway client using the anon key for auth-only calls.

        The returned client is intended to be used for a single
        sign_in_with_password / sign_up call and then discarded. Its JWT never
        leaks onto ``self.client`` — this is what prevents the concurrent-login
        race that causes RLS violations on the shared client.
        """
        if not self.anon_key:
            raise ValueError(
                "SUPABASE_ANON_KEY must be set in .env to use auth helpers"
            )
        return create_client(self.url, self.anon_key)

    def verify_credentials(self, email: str, password: str) -> Tuple[str, str]:
        """Verify an email/password pair against Supabase Auth.

        Uses a TRANSIENT anon client so the shared service-role client's auth
        header is never mutated. Returns ``(user_id, email)``.

        Raises whatever ``supabase-py`` raises on invalid credentials so
        callers can surface the auth error to the user.
        """
        tmp = self._build_anon_client()
        response = tmp.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        user = response.user
        return user.id, (user.email or email)

    def register_credentials(self, email: str, password: str) -> Tuple[str, str]:
        """Register a new Supabase Auth user via a transient anon client.

        Returns ``(user_id, email)``. Does NOT create the application-level
        row in ``public.users``; callers should follow up with
        :meth:`create_user` as they already do.
        """
        tmp = self._build_anon_client()
        response = tmp.auth.sign_up({"email": email, "password": password})
        user = response.user
        return user.id, (user.email or email)

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
            print(f"[DEBUG][search_documents] RPC start limit={limit} threshold={threshold}")
            response = self.client.rpc(
                "search_documents",
                {
                    "query_embedding": embedding,
                    "match_limit": limit,
                    "match_threshold": threshold,
                },
            ).execute()
            rows = response.data if response.data else []
            print(f"[DEBUG][search_documents] RPC returned {len(rows)} row(s)")
            if rows:
                sample = rows[0]
                print(
                    "[DEBUG][search_documents] RPC sample "
                    f"source={sample.get('source')} "
                    f"content_preview={(sample.get('content') or '')[:100]}"
                )
            return rows
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

    def delete_documents_by_doc_type(self, doc_type: str) -> int:
        """Delete all documents of a given doc_type (e.g., 'policy').
        
        Returns the number of documents deleted.
        Useful for clearing old chunks before re-ingesting with new strategy.
        """
        try:
            response = self.client.table("documents").delete().eq("doc_type", doc_type).execute()
            # Supabase DELETE returns affected row count
            count = len(response.data) if response.data else 0
            print(f"[DB] Deleted {count} documents with doc_type='{doc_type}'")
            return count
        except postgrest_exceptions.APIError as e:
            print(f"Error deleting documents: {e}")
            return 0

    def delete_documents_by_source(self, source: str) -> int:
        """Delete all documents from a specific source (e.g., 'sgbank_withdrawal_policy').
        
        Returns the number of documents deleted.
        """
        try:
            response = self.client.table("documents").delete().eq("source", source).execute()
            count = len(response.data) if response.data else 0
            print(f"[DB] Deleted {count} documents with source='{source}'")
            return count
        except postgrest_exceptions.APIError as e:
            print(f"Error deleting documents: {e}")
            return 0

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
