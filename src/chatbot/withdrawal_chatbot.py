"""
SGBank Withdrawal Assistant
RAG-powered chatbot restricted to official withdrawal policy documentation.

Designed to be instantiated once at startup and shared across requests.
Per-request state (user_id, conversation_id) is passed to chat() and
stored in a shared dict (self._ctx) so that LangGraph nodes and tool
closures can access the correct values — even when LangChain dispatches
tool calls to worker threads (where threading.local would be empty).
"""

import os
import json
import math
import time
import asyncio
import threading
import concurrent.futures
from typing import Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv
import sys
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import OpenAI

from langgraph.graph import END, StateGraph
from langchain.agents import create_agent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.supabase_client import SupabaseDB
from .sentinel_guard import SentinelGuard

_BLOCKED_RESPONSE = "Sorry I am unable to assist with that. Please feel free to ask other questions regarding withdrawal"

# NOTE: threading.local() does NOT work here because LangChain's agent
# dispatches tool calls to worker threads that don't inherit thread-local
# state.  Instead, per-request context lives in self._ctx (a plain dict
# on the chatbot instance).  With Flask sync workers each process handles
# one request at a time, so a shared dict is safe.


def _run_async(coro):
    """Run an async coroutine from synchronous (Flask/gunicorn) code.

    If no event loop is running we just use ``asyncio.run()``.  When called
    from inside an existing loop (e.g. LangGraph internals), we offload to
    a worker thread to avoid "cannot run nested event loops" errors.
    """
    try:
        asyncio.get_running_loop()
        # Already inside an event loop — run in a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Embedding-based TTL cache for RAG / policy-checker results
# ---------------------------------------------------------------------------
class _DocCache:
    """Thread-safe TTL cache that matches on cosine similarity of embeddings.

    Instead of exact string matching, this compares the question's embedding
    vector against cached embeddings.  A paraphrased question like
    "withdrawal limit" vs "what is the withdrawal limit?" will hit the cache
    as long as their cosine similarity exceeds the threshold (~0.90).

    On a cache hit the expensive steps (Supabase vector search + policy LLM)
    are skipped entirely.  The embedding itself is still computed (~100ms)
    because we need it for the similarity check, but that is cheap compared
    to the 1-3s saved.
    """

    def __init__(
        self,
        ttl: int = 300,
        max_size: int = 100,
        similarity_threshold: float = 0.90,
    ):
        # Each entry: (embedding, answer, timestamp)
        self._entries: List[tuple] = []
        self._ttl = ttl
        self._max_size = max_size
        self._threshold = similarity_threshold
        self._lock = threading.Lock()

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, embedding: List[float]) -> Optional[str]:
        """Return the cached answer whose embedding is most similar, or None."""
        with self._lock:
            now = time.time()
            # Evict expired entries
            self._entries = [
                e for e in self._entries if now - e[2] < self._ttl
            ]

            best_score = 0.0
            best_answer = None
            for cached_emb, answer, _ in self._entries:
                score = self._cosine_similarity(embedding, cached_emb)
                if score > best_score:
                    best_score = score
                    best_answer = answer

            if best_score >= self._threshold:
                return best_answer
            return None

    def put(self, embedding: List[float], answer: str) -> None:
        """Store an (embedding, answer) pair."""
        with self._lock:
            if len(self._entries) >= self._max_size:
                # Evict oldest entry
                self._entries.sort(key=lambda e: e[2])
                self._entries.pop(0)
            self._entries.append((embedding, answer, time.time()))


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

OUTPUT_CHECKER_SYS_PROMPT = """You are an output safety, relevance, and compliance checker for a public-facing bank chatbot.

Given:
- The user's message
- The assistant's draft answer

Decide whether the draft answer is safe to show.

You must enforce:
- Relevance: The final answer must address the user's message. If the user asks something unrelated to SGBank withdrawals/account support, the assistant should politely say it can only help with withdrawal/account questions.
- Do not reveal internal security procedures, monitoring rules, thresholds, or operational details.
- Do not provide instructions to bypass controls.
- Do not include any content marked INTERNAL.
- Keep the final answer customer-friendly and concise.

Actions:
- allow: safe as-is
- rewrite: unsafe/overly detailed, but can be rewritten safely
- retry: the draft is low-quality/off-topic/doesn't answer the question; request a regeneration (do not rewrite yourself)
- block: cannot be made safe without refusing

Output MUST be valid JSON with keys:
- action: "allow" | "rewrite" | "retry" | "block"
- reason: short string
- safe_answer: string (only required if action == "rewrite")""".strip()


OUTPUT_RETRY_INSTRUCTIONS = """Your previous draft answer was rejected by an output checker.

Fix it by:
- Directly answering the user's question.
- Staying strictly within SGBank withdrawal/account support scope.
- If the user's request is off-topic, respond with a brief redirect to withdrawal/account topics.
- Do not reveal internal monitoring/security procedures or bypass instructions.
- Keep it concise (3–5 sentences).""".strip()


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
- If the policy tool returns no results, honestly tell the user you could not find that information in the approved documents. Do NOT guess or use outside knowledge.
- Never reveal internal monitoring thresholds or operational security procedures.
- If the user asks something off-topic (not about withdrawals, accounts, or related bank support), politely say you can only help with withdrawal/account questions.

Response style:
- Professional, concise, 3–5 sentences.
- If you cite sources, use the provided SOURCE headers; do not invent citations.
- Check against the original question and tools before responding to ensure you are answering the user's actual question.""".strip()


POLICY_CHECKER_SYS_PROMPT = """You are the Policy Checker.

You must answer the user's question using ONLY the provided policy excerpts.

Constraints:
- If the excerpts do not contain the answer, say you cannot find it in the approved documents.
- Do not use outside knowledge.
- Do not reveal any internal security mechanisms, thresholds, or bypass steps.

Output:
- Provide a short, customer-friendly answer.
- When relevant, reference which SOURCE(s) you relied on.""".strip()


class _ChatState(TypedDict, total=False):
    user_message: str
    history: List[BaseMessage]
    trace_id: str
    blocked: bool
    block_reason: str
    answer: str
    retry_count: int
    guard_reason: str
    needs_retry: bool


class WithdrawalChatbot:
    """SGBank Withdrawal Policy Assistant (LangGraph, tool-driven).

    Instantiate once and reuse across requests.  Per-request identity is
    passed via ``chat(user_message, user_id, conversation_id)``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 400,
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
            api_key=self.api_key,
        )

        # Deterministic model for policy checker tool
        self.policy_llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            api_key=self.api_key,
        )

        # Deterministic model for output checking/sanitization
        self.output_llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=220,
            api_key=self.api_key,
        )

        self.db = db or SupabaseDB()
        self._openai_client = OpenAI(api_key=self.api_key)
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

        self.sentinel_guard = SentinelGuard()
        self._doc_cache = _DocCache(ttl=300, max_size=100)

        # Per-request context — updated at the start of each chat() call.
        # Shared dict rather than threading.local because LangChain may
        # execute tool calls in a different thread.
        self._ctx: Dict[str, Any] = {"user_id": None, "conversation_id": None, "debug": False}

        # Build tools, agent, and graph once — reused across all requests
        self._tools = self._build_tools()
        self._qa_agent = create_agent(self.llm, self._tools, system_prompt=QA_AGENT_SYS_PROMPT)
        self._graph = self._build_graph()

    def clear_history(self, user_id: str) -> str:
        """Create a fresh conversation.  Returns the new conversation_id."""
        conv = self.db.create_conversation(user_id=user_id, title="Withdrawal Bot Session")
        return conv["id"]

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

        result = _run_async(
            self.sentinel_guard.validate(
                text=user_message,
                messages=self._build_sentinel_messages(user_message),
            )
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
        db = self.db
        openai_client = self._openai_client
        policy_llm = self.policy_llm
        embedding_model = self.embedding_model
        embedding_dimensions = self.embedding_dimensions
        doc_cache = self._doc_cache
        ctx = self._ctx  # captured by reference — always sees latest values

        @tool("get_account_overview")
        def get_account_overview() -> str:
            """Return the user's balance, daily limit, and daily withdrawn (if available)."""
            uid = ctx["user_id"]
            snap = db.get_user_account_snapshot(uid)
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

            # Compute embedding first — needed for both cache lookup and RAG
            try:
                resp = openai_client.embeddings.create(
                    input=question,
                    model=embedding_model,
                    dimensions=embedding_dimensions,
                )
                query_embedding = resp.data[0].embedding
            except Exception:
                query_embedding = None

            # Check cache by cosine similarity on the embedding
            if query_embedding is not None:
                cached = doc_cache.get(query_embedding)
                if cached is not None:
                    return cached

            # Cache miss — do full RAG
            results = []
            if query_embedding is not None:
                try:
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
            answer = getattr(resp, "content", str(resp))

            # Cache the embedding → answer pair for similar future questions
            if query_embedding is not None:
                doc_cache.put(query_embedding, answer)
            return answer

        return [get_account_overview, policy_checker]

    # ---------------------------
    # Output Check (Layer 3)
    # ---------------------------
    def _sanitize_output(self, answer: str) -> str:
        text = (answer or "").strip()
        if not text:
            return "I'm sorry — I couldn't generate a response. Please try rephrasing your question."

        text = text.replace("[INTERNAL]", "").strip()

        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        return text

    def _llm_output_check(self, user_message: str, draft_answer: str) -> Dict[str, Any]:
        """Return a dict with: action, reason, (optional) safe_answer."""
        msgs = [
            SystemMessage(content=OUTPUT_CHECKER_SYS_PROMPT),
            HumanMessage(
                content=(
                    "User message:\n"
                    + (user_message or "")
                    + "\n\nDraft answer:\n"
                    + (draft_answer or "")
                )
            ),
        ]
        resp = self.output_llm.invoke(msgs)
        content = getattr(resp, "content", "") or ""

        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"action": "allow", "reason": "output_check_parse_failed"}

        try:
            data = json.loads(content[start : end + 1])
        except Exception:
            return {"action": "allow", "reason": "output_check_invalid_json"}

        action = (data.get("action") or "allow").strip().lower()
        if action not in {"allow", "rewrite", "retry", "block"}:
            action = "allow"
        out: Dict[str, Any] = {"action": action, "reason": (data.get("reason") or "").strip()}
        if action == "rewrite":
            out["safe_answer"] = (data.get("safe_answer") or "").strip()
        return out

    def _preview_text(self, text: str, limit: int = 160) -> str:
        preview = (text or "").replace("\n", " ").strip()
        return preview[:limit] + ("..." if len(preview) > limit else "")

    def _log_output_guard_event(
        self,
        event: str,
        trace_id: str,
        action: str = "",
        reason: str = "",
        retry_count: int = 0,
        user_message: str = "",
    ) -> None:
        message_preview = self._preview_text(user_message)
        print(
            "[OUTPUT_GUARD]"
            f"[trace={trace_id or 'n/a'}]"
            f"[{event}]"
            f" action='{action or 'n/a'}'"
            f" retry_count={retry_count}"
            f" reason='{reason or 'n/a'}'"
            f" user_message='{message_preview}'"
        )

    # ---------------------------
    # LangGraph
    # ---------------------------
    def _load_history(self, limit: int = 20) -> List[BaseMessage]:
        conversation_id = self._ctx["conversation_id"]
        rows = self.db.get_conversation_history(conversation_id, limit=limit)
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
                return {"blocked": True, "block_reason": "sentinel_input_blocked", "answer": _BLOCKED_RESPONSE}
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

        def qa_agent_retry_node(state: _ChatState) -> _ChatState:
            if state.get("blocked"):
                return {}
            history = state.get("history") or []
            user_message = state.get("user_message", "")
            guard_reason = (state.get("guard_reason") or "").strip()
            retry_count = int(state.get("retry_count") or 0)

            if self._ctx.get("debug", False):
                preview = (user_message or "").replace("\n", " ")
                preview = preview[:160] + ("…" if len(preview) > 160 else "")
                print(
                    f"[OUTPUT_CHECK][RETRY] Regenerating answer (attempt={retry_count}) | reason='{guard_reason}' | user_message='{preview}'"
                )

            retry_system = (
                QA_AGENT_SYS_PROMPT
                + "\n\n"
                + OUTPUT_RETRY_INSTRUCTIONS
                + (f"\n\nChecker reason: {guard_reason}" if guard_reason else "")
            )

            result = self._qa_agent.invoke(
                {"messages": [SystemMessage(content=retry_system), *history, HumanMessage(content=user_message)]}
            )
            msgs = result.get("messages") if isinstance(result, dict) else None
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                answer = getattr(last, "content", str(last))
            else:
                answer = str(result)
            return {"answer": answer}

        def output_check_node(state: _ChatState) -> _ChatState:
            if state.get("blocked"):
                return {}

            trace_id = (state.get("trace_id") or "").strip()
            user_message = state.get("user_message", "")
            draft = state.get("answer") or ""
            draft = self._sanitize_output(draft)

            verdict = self._llm_output_check(user_message=user_message, draft_answer=draft)
            action = verdict.get("action")
            guard_reason = (verdict.get("reason") or "").strip()
            retry_count = int(state.get("retry_count") or 0)

            self._log_output_guard_event(
                event="DECISION",
                trace_id=trace_id,
                action=str(action or ""),
                reason=guard_reason,
                retry_count=retry_count,
                user_message=user_message,
            )

            if action == "block":
                self._log_output_guard_event(
                    event="BLOCK",
                    trace_id=trace_id,
                    action="block",
                    reason=guard_reason,
                    retry_count=retry_count,
                    user_message=user_message,
                )
                return {"blocked": True, "needs_retry": False, "block_reason": "llm_output_blocked", "answer": _BLOCKED_RESPONSE}
            if action == "retry":
                if retry_count >= 1:
                    self._log_output_guard_event(
                        event="BLOCK",
                        trace_id=trace_id,
                        action="retry_exhausted",
                        reason=guard_reason,
                        retry_count=retry_count,
                        user_message=user_message,
                    )
                    return {"blocked": True, "needs_retry": False, "block_reason": "llm_output_retry_exhausted", "answer": _BLOCKED_RESPONSE}

                self._log_output_guard_event(
                    event="RETRY",
                    trace_id=trace_id,
                    action="retry",
                    reason=guard_reason,
                    retry_count=retry_count + 1,
                    user_message=user_message,
                )
                return {
                    "retry_count": retry_count + 1,
                    "guard_reason": guard_reason,
                    "needs_retry": True,
                }
            if action == "rewrite":
                safe_answer = (verdict.get("safe_answer") or "").strip()
                self._log_output_guard_event(
                    event="REWRITE",
                    trace_id=trace_id,
                    action="rewrite",
                    reason=guard_reason,
                    retry_count=retry_count,
                    user_message=user_message,
                )
                return {"needs_retry": False, "answer": safe_answer or draft}
            return {"needs_retry": False, "answer": draft}

        graph.add_node("load_history", load_history_node)
        graph.add_node("sentinel_input", sentinel_node)
        graph.add_node("qa_agent", qa_agent_node)
        graph.add_node("qa_agent_retry", qa_agent_retry_node)
        graph.add_node("output_check", output_check_node)

        graph.set_entry_point("load_history")
        graph.add_edge("load_history", "sentinel_input")

        def route_after_sentinel(state: _ChatState) -> str:
            return "output_check" if state.get("blocked") else "qa_agent"

        graph.add_conditional_edges("sentinel_input", route_after_sentinel, {"qa_agent": "qa_agent", "output_check": "output_check"})
        graph.add_edge("qa_agent", "output_check")

        def route_after_output_check(state: _ChatState) -> str:
            if not state.get("blocked") and state.get("needs_retry"):
                return "qa_agent_retry"
            return END

        graph.add_conditional_edges(
            "output_check",
            route_after_output_check,
            {"qa_agent_retry": "qa_agent_retry", END: END},
        )
        graph.add_edge("qa_agent_retry", "output_check")
        return graph.compile()

    # ---------------------------
    # Async helpers
    # ---------------------------
    async def _store_messages_async(
        self,
        conversation_id: str,
        user_id: str,
        user_message: str,
        answer: str,
        user_metadata: Dict[str, Any],
    ) -> None:
        """Store user + assistant messages in parallel using thread executor."""
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            loop.run_in_executor(
                None,
                lambda: self.db.add_message(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    role="user",
                    content=user_message,
                    metadata=user_metadata,
                ),
            ),
            loop.run_in_executor(
                None,
                lambda: self.db.add_message(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    role="assistant",
                    content=answer,
                    metadata={},
                ),
            ),
        )

    # ---------------------------
    # Main Chat Method
    # ---------------------------
    def chat(self, user_message: str, user_id: str, conversation_id: str, debug: bool = False) -> str:
        """Chat with the withdrawal assistant and store messages in Supabase.

        Args:
            user_message: The user's input text.
            user_id: Authenticated user ID (from Flask session).
            conversation_id: Active conversation ID.
            debug: If True, prefix the response with trace info.
        """
        # Set per-request context for LangGraph nodes and tool closures
        self._ctx["user_id"] = user_id
        self._ctx["conversation_id"] = conversation_id
        self._ctx["debug"] = debug

        try:
            trace_id = uuid4().hex[:10]
            result = self._graph.invoke({"user_message": user_message, "trace_id": trace_id})
            answer = (result or {}).get("answer") or "System error: No answer generated."
            blocked = bool((result or {}).get("blocked"))

            if blocked:
                print(
                    f"[CHAT][trace={trace_id}] blocked=True"
                    f" block_reason='{(result or {}).get('block_reason') or 'unknown'}'"
                )
                # Store only the user message with block metadata
                self.db.add_message(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    role="user",
                    content=user_message,
                    metadata={
                        "blocked": True,
                        "block_reason": (result or {}).get("block_reason"),
                    },
                )
                return answer

            # Store user + assistant messages in parallel
            _run_async(
                self._store_messages_async(
                    conversation_id,
                    user_id,
                    user_message,
                    answer,
                    {"blocked": False},
                )
            )

            if debug:
                return f"[DEBUG] trace_id={trace_id} conversation_id={conversation_id}\n\n{answer}"
            return answer

        except Exception as e:
            return f"System error: {str(e)}"
