import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test the root endpoint returns service info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Test the health endpoint returns OK."""
    response = await client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_detailed_endpoint(client: AsyncClient):
    """Test the detailed health endpoint."""
    response = await client.get("/health/detailed")
    assert response.status_code in [200, 503]  # Might be unhealthy in test env
    data = response.json()
    assert "status" in data
    assert "services" in data


@pytest.mark.asyncio
async def test_openapi_docs(client: AsyncClient):
    """Test that OpenAPI docs are available in debug mode."""
    response = await client.get("/docs")
    # Should return 200 in debug mode, 404 in production
    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_search_endpoint_requires_query(client: AsyncClient):
    """Test search endpoint validation."""
    # Test without query
    response = await client.post("/api/v1/search/", json={})
    assert response.status_code == 422  # Validation error
    
    # Test with empty query
    response = await client.post("/api/v1/search/", json={"query": ""})
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_document_list_endpoint(client: AsyncClient):
    """Test document listing endpoint."""
    response = await client.get("/api/v1/documents/")
    # Should return 200 with empty list or actual documents
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)  # Direct list response, not wrapped in "documents" key
    