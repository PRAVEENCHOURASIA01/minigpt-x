import json
import os
import pytest
from fastapi.testclient import TestClient

os.environ["MODEL_DEVICE"] = "cpu"
os.environ["MODEL_CKPT"] = "checkpoints/smoke_latest.pt"

from api.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["device"] == "cpu"


def test_info(client):
    res = client.get("/info")
    assert res.status_code == 200
    data = res.json()
    assert data["device"] == "cpu"
    assert "total_parameters" in data
    assert "config" in data
    assert data["config"]["vocab_size"] == 4096


def test_generate_non_streaming(client):
    req = {
        "prompt": "Once upon a time",
        "max_new_tokens": 10,
        "temperature": 0.0,
        "stream": False,
    }
    res = client.post("/generate", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["prompt"] == req["prompt"]
    assert len(data["text"]) > 0


def test_generate_streaming(client):
    req = {
        "prompt": "Once upon a time",
        "max_new_tokens": 10,
        "temperature": 0.0,
        "stream": True,
    }
    with client.stream("POST", "/generate", json=req) as res:
        assert res.status_code == 200
        lines = list(res.iter_lines())
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) > 0
        first_payload = json.loads(data_lines[0][6:])
        assert "token" in first_payload


def test_validation_rejection(client):
    oversized_prompt = "a" * 1001
    res = client.post("/generate", json={"prompt": oversized_prompt})
    assert res.status_code == 422

    oversized_tokens = {"prompt": "Hello", "max_new_tokens": 301}
    res2 = client.post("/generate", json=oversized_tokens)
    assert res2.status_code == 422

