#!/bin/bash

# Test script to simulate LLM tool calls to the RAG backend
BASE_URL="http://localhost:8000"

echo "Testing RAG Service LLM Tool Calls"

# 1. Add a test document
echo "1. Adding a test document..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/documents/" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "FastAPI is a modern web framework for building APIs with Python. It provides automatic API documentation, request validation, and high performance through async support.",
    "metadata": {
      "source": "test",
      "category": "framework",
      "language": "python"
    }
  }')

echo "Response: $RESPONSE"

# Wait a moment for processing
sleep 2

# 2. Search for the document
echo -e "\n2. Searching for documents about FastAPI..."
SEARCH_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is FastAPI and what are its benefits?",
    "limit": 3,
    "include_metadata": true
  }')

echo "Search Response: $SEARCH_RESPONSE" | jq '.' 2>/dev/null || echo "$SEARCH_RESPONSE"

# 3. Test vector search only
echo -e "\n3. Testing pure vector search..."
VECTOR_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/search/vector" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python web framework",
    "limit": 2
  }')

echo "Vector Search Response: $VECTOR_RESPONSE" | jq '.' 2>/dev/null || echo "$VECTOR_RESPONSE"

# 4. Test BM25 search only
echo -e "\n4. Testing pure BM25 search..."
BM25_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/search/bm25" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "FastAPI framework",
    "limit": 2
  }')

echo "BM25 Search Response: $BM25_RESPONSE" | jq '.' 2>/dev/null || echo "$BM25_RESPONSE"

# 5. Check health
echo -e "\n5. Checking service health..."
HEALTH_RESPONSE=$(curl -s "$BASE_URL/health/")
echo "Health: $HEALTH_RESPONSE" | jq '.' 2>/dev/null || echo "$HEALTH_RESPONSE"

echo -e "\n=== Test Complete ==="