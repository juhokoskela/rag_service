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
) -> int:
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


async def create_batch_job(
    documents: List[Dict[str, Any]],
    base_url: str,
    session: httpx.AsyncClient
) -> Optional[str]:
    """Create a batch job for multiple documents."""
    try:
        response = await session.post(
            f"{base_url}/api/v1/documents/batch-job",
            json=documents
        )

        if response.status_code == 202:
            job_info = response.json()
            job_id = job_info["job_id"]
            status = job_info["status"]

            print(f"Created batch job {job_id}")
            print(f"   Status: {status}")
            print(f"   Total chunks: {job_info['total_chunks']}")
            print(f"   Cached chunks: {job_info['cached_chunks']}")
            print(f"   Processing chunks: {job_info['uncached_chunks']}")

            if status == "completed":
                print(f"   Documents created: {job_info.get('documents_created', 'Processing...')}")
                return job_id
            else:
                print(f"   {job_info.get('message', 'Processing...')}")
                return job_id
        else:
            print(f"Batch job creation failed: {response.status_code}")
            try:
                error = response.json()
                print(f"Error: {error.get('detail', 'Unknown error')}")
            except:
                print(f"Error: {response.text}")
            return None

    except Exception as e:
        print(f"Error creating batch job: {e}")
        return None


async def check_batch_status(
    job_id: str,
    base_url: str,
    session: httpx.AsyncClient
) -> Dict[str, Any]:
    """Check the status of a batch job."""
    try:
        response = await session.get(f"{base_url}/api/v1/documents/batch-job/{job_id}")

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to check batch status: {response.status_code}")
            return {"status": "error"}

    except Exception as e:
        print(f"Error checking batch status: {e}")
        return {"status": "error"}


async def wait_for_batch_completion(
    job_id: str,
    base_url: str,
    session: httpx.AsyncClient,
    max_wait_minutes: int = 60
) -> bool:
    """Wait for a batch job to complete with polling."""
    print(f"Monitoring batch job {job_id}...")

    max_attempts = max_wait_minutes * 2  # Check every 30 seconds

    for attempt in range(max_attempts):
        status_info = await check_batch_status(job_id, base_url, session)

        if status_info.get("status") == "error":
            print("Error checking batch status")
            return False

        status = status_info.get("status", "unknown")

        if status == "completed":
            documents_info = status_info.get("documents", {})
            if documents_info.get("created", 0) > 0:
                print(f"✓ Batch job completed successfully!")
                print(f"   Documents created: {documents_info['created']}")
                return True
            else:
                print(f"⚠ Batch completed but no documents were stored yet")
                print(f"   This may indicate automatic processing is still running")
                # Continue polling for a bit more

        elif status in ["failed", "expired", "cancelled"]:
            print(f"✗ Batch job failed with status: {status}")
            return False

        # Show progress every 10 attempts (5 minutes)
        if attempt % 10 == 0 and attempt > 0:
            progress = status_info.get("progress", {})
            print(f"   Status: {status}")
            print(f"   Progress: {progress.get('completed', 0)}/{progress.get('total_requests', 0)} completed")

        await asyncio.sleep(30)  # Wait 30 seconds

    print(f"⚠ Batch job did not complete within {max_wait_minutes} minutes")
    print("   You can check status later with: GET /api/v1/documents/batch-job/{job_id}")
    return False


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
    max_concurrent: int = 3,
    use_batch_api: bool = True
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

    # Prepare documents for processing
    documents = []
    failed_files = []

    for file_path in files:
        try:
            content = file_path.read_text(encoding='utf-8')

            # Create metadata for this file
            file_metadata = base_metadata.copy()
            file_metadata.update({
                "filename": file_path.name,
                "filepath": str(file_path.relative_to(dir_path)),
                "extension": file_path.suffix,
                "source": "batch_script"
            })

            documents.append({
                "content": content,
                "metadata": file_metadata
            })

        except UnicodeDecodeError:
            print(f"Skipping {file_path.name} - not valid UTF-8")
            failed_files.append(file_path.name)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            failed_files.append(file_path.name)

    if not documents:
        print("No valid documents to process")
        return

    print(f"Prepared {len(documents)} documents for processing")
    if failed_files:
        print(f"Skipped {len(failed_files)} files due to errors")

    try:
        timeout = httpx.Timeout(300.0, connect=10.0)  # 5 minute timeout for batch operations
        async with httpx.AsyncClient(timeout=timeout) as session:

            # Use batch API for large batches (10+ documents)
            if use_batch_api and len(documents) >= 10:
                print(f"\nUsing batch API for {len(documents)} documents...")

                job_id = await create_batch_job(documents, base_url, session)
                if job_id:
                    success = await wait_for_batch_completion(job_id, base_url, session)
                    if success:
                        print(f"\n✓ Batch processing completed successfully!")
                        print(f"  Documents processed: {len(documents)}")
                        print(f"  Failed to read: {len(failed_files)}")
                    else:
                        print(f"\n⚠ Batch processing may not have completed successfully")
                        print(f"  Check job status: GET {base_url}/api/v1/documents/batch-job/{job_id}")
                else:
                    print("Failed to create batch job, falling back to individual processing...")
                    await process_individual_documents(documents, base_url, session, max_concurrent)

            else:
                # Use individual API for small batches
                if use_batch_api:
                    print(f"\nUsing individual API (less than 10 documents)")
                else:
                    print(f"\nUsing individual API (batch API disabled)")

                await process_individual_documents(documents, base_url, session, max_concurrent)

    except httpx.ConnectError:
        print(f"Could not connect to RAG service at {base_url}")
        print("Make sure the service is running: make up")
    except Exception as e:
        print(f"Batch processing failed: {e}")


async def process_individual_documents(
    documents: List[Dict[str, Any]],
    base_url: str,
    session: httpx.AsyncClient,
    max_concurrent: int
) -> None:
    """Process documents individually using the regular API."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_document(doc: Dict[str, Any]) -> tuple:
        async with semaphore:
            try:
                filename = doc["metadata"].get("filename", "unknown")
                print(f"Processing: {filename} ({len(doc['content'])} chars)")

                chunks = await add_document(doc["content"], doc["metadata"], base_url, session)
                return filename, chunks, True

            except Exception as e:
                filename = doc["metadata"].get("filename", "unknown")
                print(f"Error processing {filename}: {e}")
                return filename, 0, False

    # Process all documents
    tasks = [process_document(doc) for doc in documents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summary
    successful = 0
    total_chunks = 0
    failed = 0

    for result in results:
        if isinstance(result, Exception):
            failed += 1
            continue

        filename, chunks, success = result
        if success:
            successful += 1
            total_chunks += chunks
        else:
            failed += 1

    print(f"\nIndividual processing complete:")
    print(f"  Successfully processed: {successful} files")
    print(f"  Total chunks created: {total_chunks}")
    print(f"  Failed: {failed} files")


async def main():
    parser = argparse.ArgumentParser(description="Batch add documents to the RAG service")
    parser.add_argument("directory", help="Directory containing documents to process")
    parser.add_argument("--pattern", default="*", help="File pattern to match (default: *)")
    parser.add_argument("--metadata", help="JSON metadata to add to all documents")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"RAG service URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent uploads (default: 3)")
    parser.add_argument("--no-recursive", action="store_true", help="Don't search subdirectories")
    parser.add_argument("--no-batch-api", action="store_true", help="Force individual processing instead of batch API")

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
        max_concurrent=args.max_concurrent,
        use_batch_api=not args.no_batch_api
    )


if __name__ == "__main__":
    asyncio.run(main())