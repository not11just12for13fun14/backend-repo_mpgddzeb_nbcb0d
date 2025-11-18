import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="ReplyRate Backend", version="1.0.0")

# CORS: allow all for dev preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1)


class AnalyzeResponse(BaseModel):
    score: int
    reasons: List[str]
    improved: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def read_root():
    return {"message": "ReplyRate Backend is running"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    api_key = os.getenv("OPENAI_API_KEY")

    # Try real OpenAI if key present; otherwise, return a heuristic fallback
    if api_key:
        try:
            # Lazy import so the app still runs without the package installed
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)

            system = (
                "You rate outreach messages for likelihood of a positive reply. "
                "Return a JSON object with fields: score (0-100 integer), reasons (array of short strings), improved (concise improved message)."
            )
            user = (
                "Analyze this outreach message for reply probability. "
                "Be candid and specific. Message: " + msg
            )

            # Use the Responses API with JSON output
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )

            # Extract JSON text from the first output
            content = response.output_text
            import json

            parsed = json.loads(content)

            score = int(max(0, min(100, int(parsed.get("score", 68)))))
            reasons = parsed.get("reasons", []) or []
            improved = parsed.get("improved", "") or ""

            # Basic fallbacks if model returns unexpected structure
            if not isinstance(reasons, list):
                reasons = [str(reasons)] if reasons else []

            return AnalyzeResponse(score=score, reasons=reasons, improved=improved)
        except Exception:
            # Fall through to heuristic fallback on any error
            pass

    # Heuristic fallback when no key or on error
    lower = msg.lower()
    score = 75
    reasons: List[str] = []

    if len(msg) < 80:
        score -= 8
        reasons.append("Message is quite short; limited context reduces relevance.")
    if len(msg) > 600:
        score -= 10
        reasons.append("Message is long; busy recipients may skim or ignore.")
    if "calendar" in lower or "schedule" in lower:
        score -= 5
        reasons.append("Requesting time too early can depress replies.")
    if any(w in lower for w in ["free", "discount", "limited time"]):
        score -= 6
        reasons.append("Salesy phrasing may reduce trust.")
    if not any(x in lower for x in ["you", "your", "yours"]):
        score -= 7
        reasons.append("Insufficient recipient focus.")

    score = max(20, min(92, score))

    improved = (
        "Hey there — noticed your recent work on [specific initiative]. "
        "We helped a similar team achieve [clear outcome] without extra lift. "
        "Worth a quick async walkthrough or a 7‑min chat next week?"
    )

    if not reasons:
        reasons = [
            "Tighten the ask to something low‑commitment.",
            "Add one concrete, recipient‑centric benefit.",
            "Personalize with a recent post, project, or metric.",
        ]

    return AnalyzeResponse(score=score, reasons=reasons, improved=improved)


# Optional: existing database connectivity test (kept for convenience)
@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        from database import db  # type: ignore

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = getattr(db, "name", "✅ Connected")
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (optional)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Env markers
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
