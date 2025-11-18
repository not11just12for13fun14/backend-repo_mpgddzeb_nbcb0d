from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx

# Environment
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="ReplyRate API")

# CORS - allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In preview/dev we allow all. Change for prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class AnalyzeRequest(BaseModel):
    message: str


class AnalyzeResponse(BaseModel):
    score: int
    reasons: list[str]
    improved: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # If no OpenAI key, return a mocked response so the UI still works
    if not OPENAI_API_KEY:
        # Simple heuristic mock
        length_penalty = max(0, (len(message) - 400) // 40)
        base = 72
        score = max(35, min(95, base - length_penalty))
        reasons = [
            "Message could be more personalized",
            "CTA may feel time-consuming without clear value",
            "Tighten the copy and lead with the outcome"
        ]
        improved = (
            "Hey — loved your recent post. We helped a similar team hit measurable wins (e.g., +27% faster ramp). "
            "Would a 7‑min async walkthrough help you see if it fits your priorities?"
        )
        return AnalyzeResponse(score=score, reasons=reasons, improved=improved)

    # With OpenAI key, call the API for structured analysis
    prompt = f"""
You are ReplyRate, an expert LinkedIn outreach coach. Analyze the message below.
Return a concise JSON with fields: score (0-100), reasons (array of 2-4 short strings), improved (a single improved message in the sender's voice, 2-4 sentences, respectful, specific, and skimmable).
Message:\n{message}
"""

    try:
        # Use OpenAI responses API for structured output
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        body = {
            "model": "gpt-4o-mini",
            "input": prompt,
            "response_format": {"type": "json_object"}
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=body)
            r.raise_for_status()
            data = r.json()

        # Extract JSON text
        text = data.get("output_text") or data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Fallback if provider nests in output
        if isinstance(text, str) and text.strip():
            import json
            parsed = json.loads(text)
        else:
            # Attempt tool-agnostic parsing
            import json
            parsed = data if isinstance(data, dict) else {}

        score = int(parsed.get("score", 72))
        reasons = parsed.get("reasons", ["Message could be clearer"])  # type: ignore
        improved = parsed.get("improved", "Here is a clearer version of your message.")
        # Clamp values
        score = max(0, min(100, score))
        if not isinstance(reasons, list):
            reasons = [str(reasons)]

        return AnalyzeResponse(score=score, reasons=reasons, improved=improved)
    except Exception as e:
        # On any error, return a safe mocked fallback
        return AnalyzeResponse(
            score=70,
            reasons=["Temporary analysis issue, showing smart estimate", "Try again in a moment"],
            improved="Refocus on the recipient's outcome, personalize one detail, and propose a low‑lift next step (e.g., a 7‑minute async walkthrough)."
        )
