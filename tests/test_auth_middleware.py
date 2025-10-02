"""Tests for authentication middleware."""

import jwt
import pytest

from src.core.config import settings


pytestmark = pytest.mark.asyncio


async def test_shared_token_authentication(monkeypatch, client):
    monkeypatch.setattr(settings, "api_auth_token", "super-secret-token")
    monkeypatch.setattr(settings, "jwt_secret", None)

    response = await client.get("/health/")
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing Authorization header"

    response = await client.get(
        "/health/",
        headers={"Authorization": "Bearer super-secret-token"},
    )
    assert response.status_code == 200

async def test_jwt_authentication(monkeypatch, client):
    secret = "jwt-secret"
    monkeypatch.setattr(settings, "jwt_secret", secret)
    monkeypatch.setattr(settings, "api_auth_token", None)
    monkeypatch.setattr(settings, "jwt_algorithm", "HS256")
    monkeypatch.setattr(settings, "jwt_audience", None)
    monkeypatch.setattr(settings, "jwt_issuer", None)

    response = await client.get("/health/")
    assert response.status_code == 401

    invalid = await client.get(
        "/health/",
        headers={"Authorization": "Bearer invalid"},
    )
    assert invalid.status_code == 401

    token = jwt.encode({"sub": "user-123"}, secret, algorithm="HS256")
    response = await client.get(
        "/health/",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200

    # Clean up modifications
    monkeypatch.setattr(settings, "jwt_secret", None)
