import pytest
from fastapi.testclient import TestClient

from chatmod.api.app import create_app
from chatmod.pipeline import Pipeline
from chatmod.schema import Action, Decision, Label, Route
from chatmod.tier1 import KeywordScorer, Tier1
from chatmod.tier2 import FakeBackend, LLMJudge
from chatmod.tier3 import ReviewQueue


def _decision(label=Label.harassment, sev=2, text="t"):
    return Decision(message_id="m", text=text, label=label, severity=sev, confidence=0.5,
                    action=Action.hide, route=Route.tier3)


def test_queue_lifecycle():
    q = ReviewQueue(":memory:")
    low = q.enqueue(_decision(Label.scam_spam, 1, "spam"), "low conf")
    high = q.enqueue(_decision(Label.self_harm, 2, "sad"), "always reviewed")
    assert [i["id"] for i in q.list()] == [high, low]  # priority order

    item = q.claim_next("alice")
    assert item["id"] == high and item["status"] == "claimed" and item["reviewer"] == "alice"
    assert q.stats() == {"pending": 1, "claimed": 1, "resolved": 0}

    q.release(high)
    assert q.stats()["pending"] == 2

    q.resolve(high, Label.self_harm, 2, Action.hide, "alice", "sent resources")
    with pytest.raises(ValueError):
        q.resolve(high, Label.benign, 0, Action.allow)
    with pytest.raises(KeyError):
        q.resolve(999, Label.benign, 0, Action.allow)

    exported = q.export_labels()
    assert exported == [{"id": f"review-{high}", "text": "sad", "label": "self_harm", "severity": 2,
                         "source": "human_review", "llm_label": "self_harm", "reviewer": "alice"}]


@pytest.fixture
def client(cfg):
    pipe = Pipeline(cfg, Tier1(KeywordScorer(), cfg.tier1), LLMJudge(FakeBackend(), "policy"), ReviewQueue(":memory:"))
    return TestClient(create_app(pipe, cfg))


def test_moderate_endpoint(client):
    r = client.post("/moderate", json={"text": "hello chat"})
    assert r.status_code == 200
    d = r.json()
    assert d["label"] == "benign" and d["route"] == "tier1" and d["action"] == "allow"

    r = client.post("/moderate", json={"text": "FREE COINS at coinsdrop.xyz"})
    assert r.json()["label"] == "scam_spam" and r.json()["action"] == "block"


def test_review_flow_through_ui(client):
    d = client.post("/moderate", json={"text": "I want to die"}).json()
    assert d["route"] == "tier3" and d["review_id"]

    page = client.get("/review")
    assert page.status_code == 200 and "I want to die" in page.text

    nxt = client.get("/review/next?reviewer=bob", follow_redirects=False)
    assert nxt.status_code == 303 and nxt.headers["location"] == f"/review/{d['review_id']}"
    item_page = client.get(nxt.headers["location"])
    assert item_page.status_code == 200 and "self_harm" in item_page.text

    res = client.post(f"/review/{d['review_id']}/resolve",
                      data={"label": "self_harm", "severity": "2", "reviewer": "bob"}, follow_redirects=False)
    assert res.status_code == 303
    assert client.get("/api/review/stats").json()["resolved"] == 1
    exported = client.get("/api/review/export").json()
    assert exported[0]["label"] == "self_harm" and exported[0]["reviewer"] == "bob"


def test_api_resolve_validation(client):
    d = client.post("/moderate", json={"text": "I want to die"}).json()
    ok = client.post(f"/api/review/{d['review_id']}/resolve", json={"label": "benign", "severity": 0})
    assert ok.status_code == 200 and ok.json()["final_action"] == "allow"
    again = client.post(f"/api/review/{d['review_id']}/resolve", json={"label": "benign", "severity": 0})
    assert again.status_code == 409
    assert client.post("/api/review/9999/resolve", json={"label": "benign", "severity": 0}).status_code == 404


def test_playground_and_health(client):
    assert client.get("/playground").status_code == 200
    assert client.get("/healthz").json()["scorer"] == "keyword"
