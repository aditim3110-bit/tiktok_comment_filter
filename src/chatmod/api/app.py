"""FastAPI service: moderation endpoint + human review UI (Jinja2 + HTMX)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from chatmod.config import Config, load_config
from chatmod.pipeline import Pipeline
from chatmod.schema import Action, Decision, Label, Message

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class ModerateRequest(BaseModel):
    text: str
    user_id: str | None = None
    channel_id: str | None = None
    context: list[str] = []


class ResolveRequest(BaseModel):
    label: Label
    severity: int
    action: Action | None = None
    reviewer: str = "reviewer"
    notes: str = ""


def create_app(pipeline: Pipeline | None = None, cfg: Config | None = None) -> FastAPI:
    cfg = cfg or (pipeline.cfg if pipeline else load_config())
    pipe = pipeline or Pipeline.from_config(cfg)
    if pipe.queue is None:
        raise ValueError("the API needs a pipeline with a review queue")
    queue = pipe.queue
    app = FastAPI(title="chatmod", version="0.1.0")
    policy_html = cfg.policy_text()

    # ---------------- JSON API ----------------
    @app.post("/moderate", response_model=Decision)
    async def moderate(req: ModerateRequest) -> Decision:
        msg = Message(text=req.text, user_id=req.user_id, channel_id=req.channel_id, context=req.context)
        return await pipe.moderate(msg)

    @app.get("/api/review")
    def api_list(status: str = "pending", limit: int = 100):
        return queue.list(status=status or None, limit=limit)

    @app.get("/api/review/stats")
    def api_stats():
        return queue.stats()

    @app.post("/api/review/{item_id}/resolve")
    def api_resolve(item_id: int, req: ResolveRequest):
        action = req.action or cfg.action_for(req.label, req.severity)
        try:
            return queue.resolve(item_id, req.label, req.severity, action, req.reviewer, req.notes)
        except KeyError:
            raise HTTPException(404, "not found")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.get("/api/review/export")
    def api_export():
        return queue.export_labels()

    @app.get("/healthz")
    def health():
        return {"ok": True, "scorer": pipe.tier1.scorer.name, "llm": pipe.judge.name if pipe.judge else None}

    # ---------------- Reviewer UI ----------------
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/review")

    @app.get("/review", response_class=HTMLResponse, include_in_schema=False)
    def review_list(request: Request, status: str = "pending"):
        items = queue.list(status=status, limit=200)
        return TEMPLATES.TemplateResponse(request, "queue.html", {
            "items": items, "stats": queue.stats(), "status": status})

    @app.get("/review/next", include_in_schema=False)
    def review_next(reviewer: str = "reviewer"):
        item = queue.claim_next(reviewer)
        return RedirectResponse(f"/review/{item['id']}" if item else "/review?empty=1", status_code=303)

    @app.get("/review/{item_id}", response_class=HTMLResponse, include_in_schema=False)
    def review_item(request: Request, item_id: int):
        item = queue.get(item_id)
        if not item:
            raise HTTPException(404, "not found")
        decision = json.loads(item["decision_json"])
        return TEMPLATES.TemplateResponse(request, "item.html", {
            "item": item, "d": decision, "context": json.loads(item["context"]),
            "labels": [l.value for l in Label], "actions": [a.value for a in Action],
            "policy": policy_html, "stats": queue.stats()})

    @app.post("/review/{item_id}/resolve", include_in_schema=False)
    def review_resolve(item_id: int, label: str = Form(...), severity: int = Form(...),
                       action: str = Form(""), reviewer: str = Form("reviewer"), notes: str = Form("")):
        lab = Label(label)
        act = Action(action) if action else cfg.action_for(lab, severity)
        try:
            queue.resolve(item_id, lab, severity, act, reviewer, notes)
        except ValueError:
            pass  # already resolved (double click); just move on
        return RedirectResponse(f"/review/next?reviewer={reviewer}", status_code=303)

    @app.post("/review/{item_id}/release", include_in_schema=False)
    def review_release(item_id: int):
        queue.release(item_id)
        return RedirectResponse("/review", status_code=303)

    @app.get("/playground", response_class=HTMLResponse, include_in_schema=False)
    def playground(request: Request):
        return TEMPLATES.TemplateResponse(request, "playground.html", {"stats": queue.stats()})

    return app
