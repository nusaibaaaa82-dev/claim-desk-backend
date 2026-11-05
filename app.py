import importlib
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from typing import Optional
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("MODELS_CONFIG", os.path.join(BASE, "models_config.json"))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE, "claims.db"))
API_KEY = os.environ.get("API_KEY", "")
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:8080").split(",") if o.strip()]

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = json.load(f)
THRESH = CONFIG.get("thresholds", {"false": 0.65, "true": 0.35})
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class LocalHFModel:
    def __init__(self, cfg):
        self.model_path = cfg["path"]
        self.false_label = cfg.get("false_label", "false").lower()
        
        print(f"Downloading and loading model: {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval() # মডেলকে ইনফারেন্স মোডে আনা
        print(f"Model {self.model_path} loaded successfully!")

    def score(self, text: str, evidence: str = "") -> float:
        input_text = f"{text}. Context: {evidence}" if evidence else text
        try:
            # টেক্সটকে টোকেনাইজ করা
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
            
            # মডেল রান করা (কোনো গ্রেডিয়েন্ট ছাড়া)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # স্কোরগুলোকে প্রোবাবিলিটিতে রূপান্তর করা (Softmax)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            # মডেলের লেবেলগুলো খোঁজা
            label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
            
            # মিথ্যা বা ক্ষতিকর লেবেল
            false_keywords = ["false", "label_1", "hate", "toxic"]
            for kw in false_keywords:
                if kw in label2id:
                    return float(probs[label2id[kw]])
            
            # সত্য বা নিরাপদ লেবেল
            true_keywords = ["true", "label_0", "normal", "safe"]
            for kw in true_keywords:
                if kw in label2id:
                    return 1.0 - float(probs[label2id[kw]])
                    
            return 0.5
        except Exception as e:
            print(f"Local Model Error: {str(e)}")
            return 0.5


         
           

        

class PyModel:
    def __init__(self, cfg):
        mod, fn = cfg["function"].split(":")
        self.fn = getattr(importlib.import_module(mod), fn)

    def score(self, text: str, evidence: str = "") -> float:
        return float(self.fn(text))

MODELS = []
for cfg in CONFIG["models"]:
    if not cfg.get("enabled", True):
        continue
    kind = cfg.get("type", "hf")
    if kind == "hf_api":
        impl = HFAPIModel(cfg)
    else:
        impl = PyModel(cfg)
    MODELS.append((cfg["name"], float(cfg["weight"]), impl))
    print(f"loaded model: {cfg['name']} ({kind})")

def detect_lang(text: str) -> str:
    scripts = [
        (r"[ঀ-৿]", "bn"), (r"[ऀ-ॿ]", "hi"), (r"[؀-ۿ]", "ar"),
        (r"[Ѐ-ӿ]", "ru"), (r"[一-鿿]", "zh"), (r"[฀-๿]", "th"),
    ]
    for pattern, code in scripts:
        if re.search(pattern, text):
            return code
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "und"

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

with closing(db()) as c:
    c.execute(
        """CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, country TEXT, lang TEXT,
        scores TEXT, failed TEXT, p_false REAL, spread REAL, verdict TEXT,
        status TEXT DEFAULT 'pending', final_verdict TEXT, note TEXT DEFAULT '',
        video TEXT DEFAULT '', web_context TEXT DEFAULT '', created_at REAL, reviewed_at REAL)"""
    )
    c.commit()

def row(r):
    d = dict(r)
    d["scores"] = json.loads(d["scores"] or "{}")
    d["failed"] = json.loads(d["failed"] or "[]")
    return d

def verdict_for(p: float) -> str:
    return "false" if p >= THRESH["false"] else "true" if p <= THRESH["true"] else "unsure"

app = FastAPI(title="Claim Desk API")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])

def auth(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Missing or wrong API key")

class CheckIn(BaseModel):
    text: str
    country: str

class ReviewIn(BaseModel):
    action: str
    final_verdict: Optional[str] = None
    note: str = ""

class VideoIn(BaseModel):
    url: str

@app.get("/health")
def health():
    return {"ok": True, "models": [n for n, _, _ in MODELS]}

@app.post("/check", dependencies=[Depends(auth)])
def check(body: CheckIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Empty claim")
    
    web_context = ""
    try:
        results = list(DDGS().text(text, max_results=3))
        snippets = [r.get("body", "") for r in results]
        web_context = " | ".join(snippets)[:800] 
    except Exception as e:
        print("Web search failed:", e)

    scores, failed = {}, []
    for name, _, model in MODELS:
        try:
            scores[name] = max(0.0, min(1.0, model.score(text, evidence=web_context)))
        except Exception as e:
            failed.append({"model": name, "error": str(e)[:200]})
    
    if not scores:
        raise HTTPException(503, "All models failed")
    
    wsum = sum(w for n, w, _ in MODELS if n in scores)
    p_false = sum(scores[n] * w for n, w, _ in MODELS if n in scores) / wsum
    spread = max(scores.values()) - min(scores.values())
    
    with closing(db()) as c:
        cur = c.execute(
            "INSERT INTO claims (text,country,lang,scores,failed,p_false,spread,verdict,web_context,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (text, body.country, detect_lang(text), json.dumps(scores), json.dumps(failed),
             p_false, spread, verdict_for(p_false), web_context, time.time()),
        )
        c.commit()
        return row(c.execute("SELECT * FROM claims WHERE id=?", (cur.lastrowid,)).fetchone())

@app.get("/queue", dependencies=[Depends(auth)])
def queue(status: str = "pending", limit: int = 100):
    with closing(db()) as c:
        rows = c.execute(
            "SELECT * FROM claims WHERE status=? ORDER BY spread DESC, id DESC LIMIT ?", (status, limit)
        ).fetchall()
        return [row(r) for r in rows]

@app.get("/claims/{claim_id}", dependencies=[Depends(auth)])
def get_claim(claim_id: int):
    with closing(db()) as c:
        r = c.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
        if not r:
            raise HTTPException(404, "Not found")
        return row(r)

@app.post("/review/{claim_id}", dependencies=[Depends(auth)])
def review(claim_id: int, body: ReviewIn):
    with closing(db()) as c:
        r = c.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
        if not r:
            raise HTTPException(404, "Not found")
        if body.action == "approve":
            status, final = "confirmed", r["verdict"]
        elif body.action == "override":
            if body.final_verdict not in ("false", "true", "unsure"):
                raise HTTPException(400, "final_verdict must be false, true or unsure")
            status, final = "overridden", body.final_verdict
        elif body.action == "escalate":
            status, final = "escalated", None
        else:
            raise HTTPException(400, "action must be approve, override or escalate")
        c.execute(
            "UPDATE claims SET status=?, final_verdict=?, note=?, reviewed_at=? WHERE id=?",
            (status, final, body.note, time.time(), claim_id),
        )
        c.commit()
        return row(c.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone())

@app.post("/video/{claim_id}", dependencies=[Depends(auth)])
def video(claim_id: int, body: VideoIn):
    url = body.url.strip()
    if url and not re.match(r"^https?://", url, re.I):
        raise HTTPException(400, "Video link must start with http:// or https://")
    with closing(db()) as c:
        c.execute("UPDATE claims SET video=? WHERE id=?", (url, claim_id))
        c.commit()
        r = c.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
        if not r:
            raise HTTPException(404, "Not found")
        return row(r)

@app.get("/export", dependencies=[Depends(auth)])
def export():
    with closing(db()) as c:
        rows = c.execute(
            "SELECT text, lang, country, final_verdict FROM claims WHERE status IN ('confirmed','overridden')"
        ).fetchall()
        return [dict(r) for r in rows]

@app.get("/test-hf")
def test_hf():
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        return {"error": "HF_TOKEN is missing in Render Environment!"}
    
    try:
        client = InferenceClient(model="nusaiba19/xlmr-xfact-binary", token=hf_token, timeout=30)
        res = client.text_classification("The sky is green.")
        return {"status": "success", "response": res}
    except Exception as e:
        return {"error": str(e)}
