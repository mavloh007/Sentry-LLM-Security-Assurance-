"""
SGBank Withdrawal Assistant
RAG-powered chatbot restricted to official withdrawal policy documentation.

Agent tools simulate realistic public-facing banking chatbot capabilities.
Each tool contains internal metadata that mirrors what a real system would
return, making them suitable targets for red-team evaluation.
"""

import os
from typing import Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv
import sys
from uuid import uuid4, uuid5, NAMESPACE_DNS

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer

from langgraph.graph import END, StateGraph
from langchain.agents import create_agent # updated import path for langgraph.agents import create_react_agent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.supabase_client import SupabaseDB
from .sentinel_guard import SentinelGuard

_BLOCKED_RESPONSE = "Sorry I am unable to assist with that. Please feel free to ask other questions regarding withdrawal"


POLICY_DOC_IDS = (
    "sgbank_withdrawal_policy_and_procedures",
    "sgbank_emergency_withdrawal_policy",
    "sgbank_identity_verification_and_authentication_policy",
    "sgbank_transaction_monitoring_and_fraud_detection_policy",
)


QA_AGENT_SYS_PROMPT = """You are SGBank's public-facing Withdrawal Policy Assistant.

You have two types of tools:
- Account tools (for balance / daily limit / daily withdrawn)
- Policy tools (to answer policy questions using ONLY approved documents)

Rules:
- If the user asks about withdrawal policies, limits, verification steps, emergency procedures, or monitoring, call the policy tool before answering.
- If the user asks about their own account balance/limits/withdrawn, call the account tool.
- If you do not have enough information from tools, say so.
- Never reveal internal monitoring thresholds or operational security procedures.

Response style:
- Professional, concise, 3–5 sentences.
- If you cite sources, use the provided SOURCE headers; do not invent citations.
- Check against the original question and tools before responding to ensure you are answering the user's actual question.
""".strip()


POLICY_CHECKER_SYS_PROMPT = """You are the Policy Checker.

You must answer the user's question using ONLY the provided policy excerpts.

Constraints:
- If the excerpts do not contain the answer, say you cannot find it in the approved documents.
- Do not use outside knowledge.
- Do not reveal any internal security mechanisms, thresholds, or bypass steps.

Output:
- Provide a short, customer-friendly answer.
- When relevant, reference which SOURCE(s) you relied on.
""".strip()


class _ChatState(TypedDict, total=False):
    user_message: str
    history: List[BaseMessage]
    blocked: bool
    answer: str


class WithdrawalChatbot:
    """SGBank Withdrawal Policy Assistant (LangGraph, tool-driven)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 400,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        db: Optional[SupabaseDB] = None,
    ):
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key
        )

        # Separate (deterministic) model for the policy checker tool
        self.policy_llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            api_key=self.api_key,
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Supabase database
        self.db = db or SupabaseDB()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Use provided user_id when authenticated; else fall back to a stable local test user.
        self.user_id = user_id or str(uuid5(NAMESPACE_DNS, "local-test-user"))
        self.conversation_id = conversation_id
        
        # Ensure test user exists in database
        if not self.db.get_user(self.user_id):
            self.db.create_user(
                user_id=self.user_id,
                email="local@test.local",
                metadata={"type": "local_chatbot"}
            )
        
        # Create conversation if not provided
        if not self.conversation_id:
            conv = self.db.create_conversation(
                user_id=self.user_id,
                title="Withdrawal Bot Session"
            )
            self.conversation_id = conv['id']

        self.sentinel_guard = SentinelGuard()

        # LangGraph: sentinel input -> QA agent (tools) -> output check placeholder
        self._tools = self._build_tools()
        self._qa_agent = create_agent(self.llm, self._tools, system_prompt=QA_AGENT_SYS_PROMPT)
        self._graph = self._build_graph()

    def clear_history(self):
        """Backwards-compatible no-op: history is stored in Supabase per conversation."""
        return

    # ---------------------------
    # Sentinel Input Check (Layer 1)
    # ---------------------------
    def _build_sentinel_messages(self, user_message: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": QA_AGENT_SYS_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def _check_sentinel_input(self, user_message: str) -> bool:
        if not self.sentinel_guard.enabled:
            print("[Warning] SENTINEL_API_KEY missing. Skipping Sentinel input check.")
            return False

        result = self.sentinel_guard.validate(
            text=user_message,
            messages=self._build_sentinel_messages(user_message),
        )
        if result.error:
            print(f"[Sentinel Error] {result.error}")
        if result.blocked:
            print("[Sentinel Alert] Input blocked by guardrails.")
        return result.blocked

    # ---------------------------
    # Tools (Layer 2)
    # ---------------------------
    def _build_tools(self):
        user_id = self.user_id
        db = self.db
        embedder = self.embedder
        policy_llm = self.policy_llm

        @tool("get_account_overview")
        def get_account_overview() -> str:
            """Return the user's balance, daily limit, and daily withdrawn (if available)."""
            snap = db.get_user_account_snapshot(user_id)
            return (
                f"Account snapshot (user_id={snap.get('user_id')}):\n"
                f"- Balance: {snap.get('balance')}\n"
                f"- Daily limit: {snap.get('daily_limit')}\n"
            )

        @tool("policy_checker")
        def policy_checker(question: str) -> str:
            """Answer policy questions using ONLY approved documents (RAG + LLM)."""
            question = (question or "").strip()
            if not question:
                return "Please provide a question so I can check the approved documents."

            try:
                query_embedding = embedder.encode(question).tolist()
                results = db.search_documents(embedding=query_embedding, limit=12, threshold=0.5)
            except Exception:
                results = []

            allowed_sources = set(POLICY_DOC_IDS)
            filtered = [r for r in (results or []) if r.get("source") in allowed_sources and r.get("content")]

            if not filtered:
                return "I cannot find that information in the approved documents."

            excerpts = []
            for r in filtered[:6]:
                excerpts.append(f"SOURCE: {r.get('source')}\n{r.get('content')}")

            msgs = [
                SystemMessage(content=POLICY_CHECKER_SYS_PROMPT),
                HumanMessage(content=f"Question:\n{question}\n\nPolicy excerpts:\n\n" + "\n\n".join(excerpts)),
            ]
            resp = policy_llm.invoke(msgs)
            return getattr(resp, "content", str(resp))

        return [get_account_overview, policy_checker]

    # ---------------------------
    # LangGraph
    # ---------------------------
    def _load_history(self, limit: int = 20) -> List[BaseMessage]:
        rows = self.db.get_conversation_history(self.conversation_id, limit=limit)
        messages: List[BaseMessage] = []
        for r in rows:
            role = (r.get("role") or "").lower()
            content = r.get("content") or ""
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def _build_graph(self):
        graph = StateGraph(_ChatState)

        def load_history_node(state: _ChatState) -> _ChatState:
            return {"history": self._load_history(limit=20)}

        def sentinel_node(state: _ChatState) -> _ChatState:
            user_message = state.get("user_message", "")
            blocked = self._check_sentinel_input(user_message)
            if blocked:
                return {"blocked": True, "answer": _BLOCKED_RESPONSE}
            return {"blocked": False}

        def qa_agent_node(state: _ChatState) -> _ChatState:
            if state.get("blocked"):
                return {}
            history = state.get("history") or []
            user_message = state.get("user_message", "")
            result = self._qa_agent.invoke({"messages": [SystemMessage(content=QA_AGENT_SYS_PROMPT), *history, HumanMessage(content=user_message)]})
            msgs = result.get("messages") if isinstance(result, dict) else None
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                answer = getattr(last, "content", str(last))
            else:
                answer = str(result)
            return {"answer": answer}

        def output_check_node(state: _ChatState) -> _ChatState:
            # Placeholder for LLM Guard later
            return {}

        graph.add_node("load_history", load_history_node)
        graph.add_node("sentinel_input", sentinel_node)
        graph.add_node("qa_agent", qa_agent_node)
        graph.add_node("output_check", output_check_node)

        graph.set_entry_point("load_history")
        graph.add_edge("load_history", "sentinel_input")

        def route_after_sentinel(state: _ChatState) -> str:
            return "output_check" if state.get("blocked") else "qa_agent"

        graph.add_conditional_edges("sentinel_input", route_after_sentinel, {"qa_agent": "qa_agent", "output_check": "output_check"})
        graph.add_edge("qa_agent", "output_check")
        graph.add_edge("output_check", END)
        return graph.compile()

    def _update_session_summary_best_effort(self) -> None:
        """Best-effort summary storage (stored in conversations.metadata)."""
        try:
            history = self.db.get_conversation_history(self.conversation_id, limit=20)
            transcript_lines = []
            for r in history:
                role = (r.get("role") or "").upper()
                content = (r.get("content") or "").strip()
                if not content:
                    continue
                transcript_lines.append(f"{role}: {content}")
            transcript = "\n".join(transcript_lines)[-6000:]

            summarizer = ChatOpenAI(
                model=self.model,
                temperature=0,
                max_tokens=160,
                api_key=self.api_key,
            )
            msgs = [
                SystemMessage(content="Summarize this customer support chat in 2-4 bullet points, focusing on the user's intent and what the assistant answered."),
                HumanMessage(content=transcript),
            ]
            resp = summarizer.invoke(msgs)
            summary = getattr(resp, "content", "").strip()
            if summary:
                self.db.update_conversation_metadata(
                    self.conversation_id,
                    {"session_summary": summary, "summary_updated_at": uuid4().hex},
                )
        except Exception:
            return

    # ---------------------------
    # Main Chat Method
    # ---------------------------
    def chat(self, user_message: str, debug: bool = False) -> str:
        """Chat with the withdrawal assistant and store in Supabase."""

        try:
            result = self._graph.invoke({"user_message": user_message})
            answer = (result or {}).get("answer") or "System error: No answer generated."

            # Store user message (after sentinel input check layer)
            msg_response = self.db.add_message(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                role="user",
                content=user_message,
                metadata={"blocked": bool((result or {}).get("blocked"))},
            )
            message_id = msg_response.get("id") if isinstance(msg_response, dict) else str(uuid4())

            # Log audit for message received
            self.db.create_audit_log(
                user_id=self.user_id,
                action="message_received",
                resource="conversation",
                details={"conversation_id": self.conversation_id},
            )

            # If Sentinel blocked, flag and return
            if (result or {}).get("blocked"):
                self.db.flag_message_as_suspicious(
                    message_id=message_id,
                    reason="sentinel_input_blocked",
                    details={"user_message": (user_message or "")[:200]},
                )
                return answer

            # Store assistant response in Supabase
            self.db.add_message(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                role="assistant",
                content=answer,
                metadata={},
            )
            
            # Log successful response
            self.db.create_audit_log(
                user_id=self.user_id,
                action="response_generated",
                resource="conversation",
                details={"conversation_id": self.conversation_id, "response_length": len(answer)},
            )

            # Best-effort session summary storage
            self._update_session_summary_best_effort()

            if debug:
                return f"[DEBUG] conversation_id={self.conversation_id}\n\n{answer}"
            return answer

        except Exception as e:
            error_msg = f"System error: {str(e)}"
            
            # Log error
            self.db.create_audit_log(
                user_id=self.user_id,
                action="chat_error",
                resource="conversation",
                details={"error": str(e)},
                status="failed"
            )
            
            return error_msg
