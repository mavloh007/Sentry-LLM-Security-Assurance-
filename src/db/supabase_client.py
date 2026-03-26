"""
Supabase Client for SGBank Withdrawal Chatbot
Handles authentication, conversations, embeddings, and audit logging
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
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
        """Search documents using vector similarity (pgvector)"""
        try:
            # Try using RPC function first
            try:
                response = self.client.rpc(
                    "search_documents",
                    {
                        "query_embedding": embedding,
                        "match_limit": limit,
                        "match_threshold": threshold,
                    },
                ).execute()
                if response.data:
                    return response.data
            except Exception as rpc_error:
                print(f"[DEBUG] RPC function failed: {rpc_error}")
                print(f"[DEBUG] Falling back to direct vector search...")
            
            # Fallback: Direct vector search using PostgreSQL operators
            # Convert embedding to string format pgvector expects
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            
            response = (
                self.client.table("documents")
                .select("id, content, source, metadata, embedding")
                .execute()
            )
            
            if not response.data:
                return []
            
            # Calculate similarity scores locally
            import math
            results = []
            for doc in response.data:
                if doc.get('embedding'):
                    # Calculate cosine similarity
                    doc_emb = doc.get('embedding')
                    if isinstance(doc_emb, str):
                        # Parse if it's a string
                        import json
                        try:
                            doc_emb = json.loads(doc_emb)
                        except:
                            continue
                    
                    # Cosine similarity
                    dot_product = sum(a * b for a, b in zip(embedding, doc_emb))
                    norm_q = math.sqrt(sum(a * a for a in embedding))
                    norm_d = math.sqrt(sum(a * a for a in doc_emb))
                    
                    if norm_q > 0 and norm_d > 0:
                        similarity = dot_product / (norm_q * norm_d)
                    else:
                        similarity = 0
                    
                    # Convert to distance for threshold comparison
                    distance = 1 - similarity
                    
                    if distance < (1 - threshold):  # If similarity > threshold
                        results.append({
                            "id": doc.get("id"),
                            "content": doc.get("content"),
                            "source": doc.get("source"),
                            "similarity": similarity,
                            "distance": distance
                        })
            
            # Sort by similarity (descending) and return top K
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            import traceback
            traceback.print_exc()
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

    # ==================== AUDIT LOGGING ====================

    def create_audit_log(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: Optional[Dict] = None,
        status: str = "success",
    ) -> Dict[str, Any]:
        """Create an audit log entry"""
        try:
            log_data = {
                "id": str(uuid4()),
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "details": details or {},
                "status": status,
                "created_at": datetime.utcnow().isoformat(),
            }
            response = self.client.table("audit_logs").insert(log_data).execute()
            return response.data[0] if response.data else log_data
        except postgrest_exceptions.APIError as e:
            print(f"Error creating audit log: {e}")
            return log_data

    def get_user_audit_logs(
        self,
        user_id: str,
        limit: int = 100,
        action: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit logs for a user"""
        try:
            query = self.client.table("audit_logs").select("*").eq("user_id", user_id)
            if action:
                query = query.eq("action", action)
            response = query.order("created_at", desc=True).limit(limit).execute()
            return response.data if response.data else []
        except postgrest_exceptions.APIError as e:
            print(f"Error fetching audit logs: {e}")
            return []

    # ==================== SESSION MANAGEMENT ====================

    def create_session(
        self,
        user_id: str,
        conversation_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        expires_in_hours: int = 24,
    ) -> Dict[str, Any]:
        """Create a user session"""
        try:
            session_data = {
                "id": str(uuid4()),
                "user_id": user_id,
                "conversation_id": conversation_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(hours=expires_in_hours)).isoformat(),
                "is_active": True,
            }
            response = self.client.table("sessions").insert(session_data).execute()
            return response.data[0] if response.data else session_data
        except postgrest_exceptions.APIError as e:
            print(f"Error creating session: {e}")
            raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        try:
            response = (
                self.client.table("sessions")
                .select("*")
                .eq("id", session_id)
                .single()
                .execute()
            )
            return response.data if response.data else None
        except postgrest_exceptions.APIError as e:
            print(f"Error fetching session: {e}")
            return None

    def end_session(self, session_id: str) -> bool:
        """End a session"""
        try:
            response = (
                self.client.table("sessions")
                .update({"is_active": False})
                .eq("id", session_id)
                .execute()
            )
            return bool(response.data)
        except postgrest_exceptions.APIError as e:
            print(f"Error ending session: {e}")
            return False

    # ==================== CONVERSATION SAFETY & METADATA ====================

    def flag_message_as_suspicious(
        self,
        message_id: str,
        reason: str,
        details: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Flag a message for review (security audit)"""
        try:
            flag_data = {
                "id": str(uuid4()),
                "message_id": message_id,
                "reason": reason,
                "details": details or {},
                "flagged_at": datetime.utcnow().isoformat(),
                "reviewed": False,
            }
            response = self.client.table("message_flags").insert(flag_data).execute()
            return response.data[0] if response.data else flag_data
        except postgrest_exceptions.APIError as e:
            print(f"Error flagging message: {e}")
            raise

    def get_flagged_messages(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get flagged messages for review"""
        try:
            query = self.client.table("message_flags").select("*").eq("reviewed", False)
            if user_id:
                # Join with messages to filter by user_id
                query = self.client.table("message_flags").select(
                    "*, messages(user_id)"
                ).eq("messages.user_id", user_id)
            response = query.order("flagged_at", desc=True).execute()
            return response.data if response.data else []
        except postgrest_exceptions.APIError as e:
            print(f"Error fetching flagged messages: {e}")
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
            # Get message count
            messages = self.client.table("messages").select("id").eq("user_id", user_id).execute()
            message_count = len(messages.data) if messages.data else 0

            # Get conversation count
            conversations = (
                self.client.table("conversations").select("id").eq("user_id", user_id).execute()
            )
            conversation_count = len(conversations.data) if conversations.data else 0

            # Get audit log count
            audit_logs = (
                self.client.table("audit_logs").select("id").eq("user_id", user_id).execute()
            )
            audit_count = len(audit_logs.data) if audit_logs.data else 0

            return {
                "user_id": user_id,
                "message_count": message_count,
                "conversation_count": conversation_count,
                "audit_count": audit_count,
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
