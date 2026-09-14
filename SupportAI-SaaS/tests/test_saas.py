import os
import shutil

import pytest
from fastapi.testclient import TestClient

# Use an isolated test database/tenant-data dir so tests never touch real data
os.environ.setdefault("TESTING", "1")

from app.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True, scope="module")
def cleanup_tenant_data():
    yield
    # best-effort cleanup after the test module runs
    tenant_data_dir = os.path.join(os.path.dirname(__file__), "..", "tenant_data")
    for name in os.listdir(tenant_data_dir):
        if name.startswith("tenant_"):
            shutil.rmtree(os.path.join(tenant_data_dir, name), ignore_errors=True)


def test_signup_returns_api_key():
    r = client.post("/signup", json={"company_name": "Test Signup Co"})
    assert r.status_code == 200
    data = r.json()
    assert data["api_key"].startswith("sk_")
    assert data["company_name"] == "Test Signup Co"


def test_classify_without_api_key_is_rejected():
    r = client.post("/classify", json={"text": "hello"})
    assert r.status_code == 401


def test_classify_with_invalid_api_key_is_rejected():
    r = client.post("/classify", json={"text": "hello"}, headers={"X-API-Key": "sk_invalid"})
    assert r.status_code == 401


def test_classify_before_training_returns_400():
    signup = client.post("/signup", json={"company_name": "Untrained Co"}).json()
    r = client.post("/classify", json={"text": "hello"}, headers={"X-API-Key": signup["api_key"]})
    assert r.status_code == 400


def test_train_with_demo_data_succeeds():
    signup = client.post("/signup", json={"company_name": "Demo Data Co"}).json()
    r = client.post("/train", json={"use_demo_data": True}, headers={"X-API-Key": signup["api_key"]})
    assert r.status_code == 200
    data = r.json()
    assert data["n_intents"] == 8
    assert 0 <= data["test_accuracy"] <= 1


def test_train_with_custom_data_succeeds():
    signup = client.post("/signup", json={"company_name": "Pizza Test Co"}).json()
    r = client.post("/train", json={
        "texts": ["order a pizza", "I want pepperoni", "cancel my order", "cancel please",
                  "where is my food", "delivery is late"],
        "labels": ["order", "order", "cancel", "cancel", "delivery", "delivery"],
    }, headers={"X-API-Key": signup["api_key"]})
    assert r.status_code == 200
    data = r.json()
    assert data["n_intents"] == 3
    assert data["used_demo_data"] is False


def test_train_with_mismatched_arrays_returns_422():
    signup = client.post("/signup", json={"company_name": "Bad Data Co"}).json()
    r = client.post("/train", json={
        "texts": ["a", "b"], "labels": ["x"],
    }, headers={"X-API-Key": signup["api_key"]})
    assert r.status_code == 422


def test_two_tenants_are_fully_isolated():
    signup_a = client.post("/signup", json={"company_name": "Tenant A"}).json()
    signup_b = client.post("/signup", json={"company_name": "Tenant B"}).json()

    client.post("/train", json={"use_demo_data": True}, headers={"X-API-Key": signup_a["api_key"]})
    client.post("/train", json={
        "texts": ["order a pizza", "cancel my order", "delivery is late", "menu options please"],
        "labels": ["order", "cancel", "delivery", "menu"],
    }, headers={"X-API-Key": signup_b["api_key"]})

    client.post("/classify", json={"text": "why was I charged twice"}, headers={"X-API-Key": signup_a["api_key"]})
    client.post("/classify", json={"text": "order a pizza"}, headers={"X-API-Key": signup_b["api_key"]})

    usage_a = client.get("/usage", headers={"X-API-Key": signup_a["api_key"]}).json()
    usage_b = client.get("/usage", headers={"X-API-Key": signup_b["api_key"]}).json()

    assert usage_a["total_classifications"] == 1
    assert usage_b["total_classifications"] == 1
    assert usage_a["company_name"] == "Tenant A"
    assert usage_b["company_name"] == "Tenant B"


def test_usage_cost_calculation():
    signup = client.post("/signup", json={"company_name": "Billing Test Co"}).json()
    client.post("/train", json={"use_demo_data": True}, headers={"X-API-Key": signup["api_key"]})
    client.post("/classify", json={"text": "why was I charged twice"}, headers={"X-API-Key": signup["api_key"]})

    usage = client.get("/usage", headers={"X-API-Key": signup["api_key"]}).json()
    assert usage["estimated_cost_usd"] >= 0
    assert usage["cost_if_all_llm_usd"] >= usage["estimated_cost_usd"]
