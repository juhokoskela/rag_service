import pytest
import pytest_asyncio

@pytest.mark.asyncio
async def test_full_document_lifecycle(client):
# 1. Create document
    doc_data = {
        "content": "This is a test document for E2E testing...",
        "metadata": {"source": "test", "type": "article"}
    }
    create_response = await client.post("/api/v1/documents/", json=doc_data)
    assert create_response.status_code == 201
    documents = create_response.json()
    assert len(documents) > 0
    doc_id = documents[0]["id"]  # Get ID from first chunk

    # 2. Search for document
    search_response = await client.post("/api/v1/search/", json={
        "query": "test document",
        "limit": 5
    })
    assert search_response.status_code == 200
    results = search_response.json()["results"]
    assert len(results) > 0

    # 3. Verify document in results
    doc_found = any(result["document"]["id"] == doc_id for result in results)
    assert doc_found

    # 4. Delete document
    delete_response = await client.delete(f"/api/v1/documents/{doc_id}")
    assert delete_response.status_code == 204
