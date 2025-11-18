import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a known .env path next to this file
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(find_dotenv(_ENV_PATH) or _ENV_PATH, override=True)

app = FastAPI(title="ReplyRate Backend", version="1.1.1")

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
    # Expose minimal signal that env was picked up (without leaking secrets)
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {"status": "ok", "openai": "set" if has_key else "missing"}


@app.get("/")
def read_root():
    return {"message": "ReplyRate Backend is running"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    api_key = os.getenv("OPENAI_API_KEY")

    # If an API key is present, we always use OpenAI for ALL fields (score, reasons, improved).
    # If OpenAI errors, we return a 502 to signal the client instead of falling back.
    if api_key:
        try:
            from openai import OpenAI  # type: ignore
            import json

            client = OpenAI(api_key=api_key)

            system = (
                "You evaluate cold outreach for likelihood of a positive reply. "
                "Respond ONLY as a strict JSON object with fields: "
                "score (integer 0-100), reasons (array of 2-6 short strings), improved (string). "
                "Be specific, concise, and actionable."
            )
            user = (
                "Analyze this outreach message and produce a JSON object as specified.\n\n"
                f"Message: " + msg
            )

            # Use Chat Completions API for maximum compatibility; request JSON output
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            content = (completion.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("Empty response from OpenAI")

            parsed = json.loads(content)

            if not isinstance(parsed, dict):
                raise ValueError("OpenAI response not a JSON object")

            # Validate score
            raw_score = parsed.get("score")
            if raw_score is None:
                raise ValueError("Missing 'score' in OpenAI response")
            try:
                score = int(raw_score)
            except Exception:
                raise ValueError("Non-integer 'score' in OpenAI response")
            score = max(0, min(100, score))

            # Validate reasons
            reasons = parsed.get("reasons")
            if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
                raise ValueError("'reasons' must be an array of strings")

            # Validate improved
            improved = parsed.get("improved")
            if not isinstance(improved, str) or not improved.strip():
                raise ValueError("'improved' must be a non-empty string")

            return AnalyzeResponse(score=score, reasons=reasons, improved=improved)
        except Exception as e:
            # With a key present, we do NOT fall back to heuristic; surface an error
            raise HTTPException(status_code=502, detail=f"OpenAI_error: {str(e)}")

    # No API key present: provide heuristic fallback so the UI remains usable
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


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
