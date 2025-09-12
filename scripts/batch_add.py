#!/usr/bin/env python3
"""
Batch document processing for the RAG service.

Usage:
    python scripts/batch_add.py /path/to/documents/
    python scripts/batch_add.py /path/to/documents/ --pattern "*.md"
    python scripts/batch_add.py /path/to/documents/ --metadata '{"source": "docs"}'
"""

import argparse
import json
import sys
import asyncio
from pathlib import Path
import httpx
from typing import Dict, Any, Optional, List
import fnmatch


DEFAULT_BASE_URL = "http://localhost:8000"
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.rst', '.py', '.js', '.html', '.xml', '.json', '.csv', '.log'}


async def add_document(
    content: str,
    metadata: Dict[str, Any],
    base_url: str,
    session: httpx.AsyncClient
) -> bool:
    """Add a single document to the RAG service."""
    payload = {
        "content": content,
        "metadata": metadata
    }
    
    try:
        response = await session.post(
            f"{base_url}/api/v1/documents/",
            json=payload
        )
        
        if response.status_code == 201:
            documents = response.json()
            return len(documents)
        else:
            print(f"   Failed: {response.status_code}")
            try:
                error = response.json()
                print(f"   Error: {error.get('detail', 'Unknown error')}")
            except:
                print(f"   Error: {response.text}")
            return 0
            
    except Exception as e:
        print(f"   Error: {e}")
        return 0


def find_documents(
    directory: Path,
    pattern: str = "*",
    recursive: bool = True
) -> List[Path]:
    """Find all document files in a directory."""
    files = []
    
    if recursive:
        search_pattern = f"**/{pattern}"
        paths = directory.glob(search_pattern)
    else:
        paths = directory.glob(pattern)
    
    for path in paths:
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    
    return sorted(files)


async def process_batch(
    directory: str,
    pattern: str = "*",
    base_metadata: Optional[Dict[str, Any]] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_concurrent: int = 3
) -> None:
    """Process all documents in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Directory not found: {directory}")
        return
    
    if not dir_path.is_dir():
        print(f"Path is not a directory: {directory}")
        return
    
    # Find all files
    files = find_documents(dir_path, pattern)
    if not files:
        print(f"No supported files found in {directory}")
        print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        return
    
    print(f"Found {len(files)} files to process")
    
    if base_metadata is None:
        base_metadata = {}
    
    # Process files with controlled concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_file(file_path: Path, session: httpx.AsyncClient) -> tuple:
        async with semaphore:
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Create metadata for this file
                file_metadata = base_metadata.copy()
                file_metadata.update({
                    "filename": file_path.name,
                    "filepath": str(file_path.relative_to(dir_path)),
                    "extension": file_path.suffix,
                    "source": "batch"
                })
                
                print(f"Processing: {file_path.relative_to(dir_path)} ({len(content)} chars)")
                
                chunks = await add_document(content, file_metadata, base_url, session)
                return file_path, chunks, True
                
            except UnicodeDecodeError:
                print(f"Skipping {file_path.name} - not valid UTF-8")
                return file_path, 0, False
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                return file_path, 0, False
    
    # Process all files
    try:
        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as session:
            tasks = [process_file(file_path, session) for file_path in files]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Summary
        successful = 0
        total_chunks = 0
        failed = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                continue
                
            file_path, chunks, success = result
            if success:
                successful += 1
                total_chunks += chunks
            else:
                failed += 1
        
        print(f"\nBatch processing complete:")
        print(f"  Successfully processed: {successful} files")
        print(f"  Total chunks created: {total_chunks}")
        print(f"  Failed: {failed} files")
        
    except httpx.ConnectError:
        print(f"Could not connect to RAG service at {base_url}")
        print("Make sure the service is running: make up")
    except Exception as e:
        print(f"Batch processing failed: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Batch add documents to the RAG service")
    parser.add_argument("directory", help="Directory containing documents to process")
    parser.add_argument("--pattern", default="*", help="File pattern to match (default: *)")
    parser.add_argument("--metadata", help="JSON metadata to add to all documents")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"RAG service URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent uploads (default: 3)")
    parser.add_argument("--no-recursive", action="store_true", help="Don't search subdirectories")
    
    args = parser.parse_args()
    
    # Parse metadata
    base_metadata = {}
    if args.metadata:
        try:
            base_metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Invalid JSON in --metadata")
            sys.exit(1)
    
    await process_batch(
        directory=args.directory,
        pattern=args.pattern,
        base_metadata=base_metadata,
        base_url=args.base_url,
        max_concurrent=args.max_concurrent
    )


if __name__ == "__main__":
    asyncio.run(main())