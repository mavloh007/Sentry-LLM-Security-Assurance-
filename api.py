import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.db.supabase_client import SupabaseDB
from src.chatbot.withdrawal_chatbot import WithdrawalChatbot

load_dotenv()

app = FastAPI(title="SGBank Withdrawal Assistant API")

class ChatRequest(BaseModel):
    message: str
    debug: bool = False

class ChatResponse(BaseModel):
    response: str


class ResetResponse(BaseModel):
    status: str


class SearchRequest(BaseModel):
    query: str
    n_results: int = 3


class SearchResponse(BaseModel):
    documents: list[str]

@app.on_event("startup")
def startup() -> None:
    db = SupabaseDB()
    app.state.bot = WithdrawalChatbot(db=db)
    app.state.db = db

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    bot = app.state.bot
    try:
        return ChatResponse(response=bot.chat(req.message, debug=req.debug))
    except Exception as e:
        # Keep error surface simple for now (you can refine later)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", response_model=ResetResponse)
def reset():
    """Clear server-side chat history.

    Useful for orchestrators (Airflow) to ensure each scenario starts fresh.
    """
    bot = getattr(app.state, "bot", None)
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    try:
        bot.clear_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ResetResponse(status="ok")


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Search the policy vector store and return matching text chunks."""

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    if req.n_results < 1 or req.n_results > 20:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 20")

    try:
        bot = getattr(app.state, "bot", None)
        db = getattr(app.state, "db", None)
        if bot is None or db is None:
            raise HTTPException(status_code=503, detail="Bot/DB not initialized")

        embedding_model = getattr(bot, "embedding_model", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        embedding_dimensions = getattr(bot, "embedding_dimensions", int(os.getenv("EMBEDDING_DIMENSIONS", "384")))
        resp = bot._openai_client.embeddings.create(
            input=req.query,
            model=embedding_model,
            dimensions=embedding_dimensions,
        )
        query_embedding = resp.data[0].embedding
        results = db.search_documents(embedding=query_embedding, limit=req.n_results, threshold=0.5)
        documents = [str(r.get("content")) for r in (results or []) if r.get("content")]
        return SearchResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))