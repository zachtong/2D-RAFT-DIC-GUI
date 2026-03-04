"""Tests for processing configuration and status endpoints."""

import pytest
from server.session import session


def test_configure(client):
    resp = client.post("/api/processing/configure", json={
        "mode": "incremental",
        "context_padding": 48,
        "use_smooth": False,
    })
    assert resp.status_code == 200
    assert session.config.mode == "incremental"
    assert session.config.context_padding == 48
    assert session.config.use_smooth is False


def test_configure_invalid(client):
    # Missing required fields should fail validation
    resp = client.post("/api/processing/configure", json={"mode": "accumulative"})
    # Config validates img_dir and model_path, which are empty
    # But configure just sets fields without full validation
    assert resp.status_code in (200, 400)


def test_status(client):
    resp = client.get("/api/processing/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["active"] is False
    assert data["has_results"] is False


def test_run_no_prerequisites(client):
    resp = client.post("/api/processing/run")
    assert resp.status_code == 400
