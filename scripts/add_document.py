#!/usr/bin/env python3
"""
Simple CLI tool for adding documents to the RAG service.

Usage:
    python scripts/add_document.py "Your document content here"
    python scripts/add_document.py --file document.txt
    python scripts/add_document.py --file document.txt --metadata '{"source": "file", "author": "John"}'
    python scripts/add_document.py --url "https://example.com/doc"
"""

import argparse
import json
import sys
from pathlib import Path
import httpx
import asyncio
from typing import Optional, Dict, Any


DEFAULT_BASE_URL = "http://localhost:8000"


async def add_document(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    base_url: str = DEFAULT_BASE_URL
) -> bool:
    """Add a document to the RAG service."""
    if metadata is None:
        metadata = {}
    
    payload = {
        "content": content,
        "metadata": metadata
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/api/v1/documents/",
                json=payload
            )
            
            if response.status_code == 201:
                documents = response.json()
                print(f"Successfully added document!")
                print(f"Created {len(documents)} chunks")
                for i, doc in enumerate(documents[:3]):  # Show first 3 chunks
                    preview = doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"]
                    print(f"   Chunk {i+1}: {preview}")
                if len(documents) > 3:
                    print(f"   ... and {len(documents) - 3} more chunks")
                return True
            else:
                print(f"Failed to add document: {response.status_code}")
                try:
                    error = response.json()
                    print(f"   Error: {error.get('detail', 'Unknown error')}")
                except:
                    print(f"   Error: {response.text}")
                return False
                
    except httpx.ConnectError:
        print(f"Could not connect to RAG service at {base_url}")
        print("   Make sure the service is running: make up")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


async def add_document_from_file(
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    base_url: str = DEFAULT_BASE_URL
) -> bool:
    """Add a document from a file."""
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {file_path}")
            return False
        
        content = path.read_text(encoding='utf-8')
        
        # Add filename to metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "filename": path.name,
            "source": "file"
        })
        
        print(f"Reading from {file_path} ({len(content)} characters)")
        return await add_document(content, metadata, base_url)
        
    except UnicodeDecodeError:
        print(f"Could not read file {file_path} - not a valid text file")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False


async def add_document_from_url(
    url: str,
    metadata: Optional[Dict[str, Any]] = None,
    base_url: str = DEFAULT_BASE_URL
) -> bool:
    """Add a document from a URL."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            content = response.text
            
            # Add URL to metadata
            if metadata is None:
                metadata = {}
            metadata.update({
                "url": url,
                "source": "web"
            })
            
            print(f"Fetched from {url} ({len(content)} characters)")
            return await add_document(content, metadata, base_url)
            
    except httpx.HTTPError as e:
        print(f"Error fetching URL: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Add documents to the RAG service")
    parser.add_argument("content", nargs="?", help="Document content (if not using --file or --url)")
    parser.add_argument("--file", help="Path to text file to add")
    parser.add_argument("--url", help="URL to fetch and add as document")
    parser.add_argument("--metadata", help="JSON metadata for the document")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"RAG service URL (default: {DEFAULT_BASE_URL})")
    
    args = parser.parse_args()
    
    # Parse metadata
    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Invalid JSON in --metadata")
            sys.exit(1)
    
    # Determine input source
    if args.file:
        success = await add_document_from_file(args.file, metadata, args.base_url)
    elif args.url:
        success = await add_document_from_url(args.url, metadata, args.base_url)
    elif args.content:
        success = await add_document(args.content, metadata, args.base_url)
    else:
        print("Must provide content, --file, or --url")
        parser.print_help()
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())