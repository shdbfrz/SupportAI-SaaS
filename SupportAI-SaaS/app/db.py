"""
Multi-tenant store: each signed-up company (tenant) gets an API key,
isolated model storage, and a running usage log. SQLite is enough for a
demo SaaS — a real deployment would swap this for Postgres, but the
schema/queries would look almost identical.
"""
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "tenant_data", "saas.db")


def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tenants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            created_at REAL NOT NULL,
            model_trained INTEGER DEFAULT 0,
            plan TEXT DEFAULT 'pay_as_you_go'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            route TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            FOREIGN KEY(tenant_id) REFERENCES tenants(id)
        )
    """)
    conn.commit()
    conn.close()


_init_db()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_tenant(company_name: str) -> dict:
    api_key = "sk_" + secrets.token_hex(20)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO tenants (company_name, api_key, created_at) VALUES (?, ?, ?)",
            (company_name, api_key, time.time()),
        )
        tenant_id = cur.lastrowid
    tenant_dir = tenant_model_dir(tenant_id)
    os.makedirs(tenant_dir, exist_ok=True)
    return {"tenant_id": tenant_id, "company_name": company_name, "api_key": api_key}


def get_tenant_by_api_key(api_key: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM tenants WHERE api_key = ?", (api_key,)).fetchone()
    return dict(row) if row else None


def mark_trained(tenant_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE tenants SET model_trained = 1 WHERE id = ?", (tenant_id,))


def log_usage(tenant_id: int, route: str, latency_ms: float):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO usage_log (tenant_id, timestamp, route, latency_ms) VALUES (?, ?, ?, ?)",
            (tenant_id, time.time(), route, latency_ms),
        )


def get_usage_summary(tenant_id: int) -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT route, COUNT(*) as cnt, AVG(latency_ms) as avg_latency FROM usage_log WHERE tenant_id = ? GROUP BY route",
            (tenant_id,),
        ).fetchall()
    summary = {"fast_classifier": {"count": 0, "avg_latency_ms": 0},
               "llm_fallback": {"count": 0, "avg_latency_ms": 0}}
    for row in rows:
        summary[row["route"]] = {"count": row["cnt"], "avg_latency_ms": round(row["avg_latency"], 3)}
    return summary


def tenant_model_dir(tenant_id: int) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "tenant_data", f"tenant_{tenant_id}")
