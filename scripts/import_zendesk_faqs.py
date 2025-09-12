#!/usr/bin/env python3
"""
Import FAQ articles from Zendesk into the RAG service.

Usage:
    python scripts/import_zendesk_faqs.py
    python scripts/import_zendesk_faqs.py --category-id 123456 --service-type tmi
    python scripts/import_zendesk_faqs.py --dry-run
"""

import asyncio
import argparse
import json
import logging
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiohttp
import httpx
from bs4 import BeautifulSoup

# Add the src directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZendeskClient:
    """Simple Zendesk API client for fetching articles."""
    
    def __init__(self, subdomain: str, email: str, token: str):
        self.subdomain = subdomain
        self.auth = (f"{email}/token", token)
        self.base_url = f"https://{subdomain}.zendesk.com/api/v2"
    
    async def fetch_articles_for_category(
        self, 
        session: aiohttp.ClientSession, 
        category_id: int,
        service_type: str
    ) -> List[Dict[str, Any]]:
        """Fetch all articles from a specific category."""
        articles = []
        url = f"{self.base_url}/help_center/categories/{category_id}/articles.json"
        
        while url:
            logger.info(f"Fetching articles for {service_type} from {url}")
            
            async with session.get(url, auth=aiohttp.BasicAuth(*self.auth)) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch articles: {response.status}")
                    break
                
                data = await response.json()
                articles.extend(data.get('articles', []))
                url = data.get('next_page')
        
        logger.info(f"Fetched {len(articles)} articles for {service_type}")
        return articles


class ZendeskProcessor:
    """Process and clean Zendesk article content."""
    
    @staticmethod
    def clean_html_content(html_content: str) -> str:
        """Clean HTML content and extract readable text."""
        if not html_content:
            return ""
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces to single
        text = text.strip()
        
        return text
    
    @staticmethod
    def is_valid_article(article: Dict[str, Any]) -> bool:
        """Check if article is valid for import."""
        return (
            article.get('draft', True) is False and  # Published articles only
            article.get('title', '').strip() != '' and
            article.get('body', '').strip() != ''
        )
    
    @staticmethod
    def extract_metadata(article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from Zendesk article."""
        return {
            'source': 'zendesk',
            'article_id': str(article.get('id', '')),
            'title': article.get('title', ''),
            'locale': article.get('locale', 'fi'),
            'created_at': article.get('created_at', ''),
            'updated_at': article.get('updated_at', ''),
            'author_id': str(article.get('author_id', '')),
            'section_id': str(article.get('section_id', '')),
            'url': article.get('html_url', ''),
            'vote_sum': article.get('vote_sum', 0),
            'vote_count': article.get('vote_count', 0)
        }


class FAQImporter:
    """Main FAQ import service."""
    
    def __init__(self, zendesk_client: ZendeskClient, rag_service_url: str):
        self.zendesk_client = zendesk_client
        self.rag_service_url = rag_service_url
        self.processor = ZendeskProcessor()
    
    async def import_all_faqs(self, category_mapping: Dict[str, int], use_batch_api: bool = True):
        """Import all FAQ articles from configured categories using batch processing."""
        logger.info(f"Starting Zendesk FAQ import for categories: {category_mapping}")
        
        try:
            articles = await self._fetch_all_articles(category_mapping)
            
            # Prepare all articles for batch processing
            all_documents = []
            article_info = []  # Keep track of original articles for logging
            
            for service_type, category_articles in articles.items():
                for article_data in category_articles:
                    document_data = self._prepare_article_for_batch(article_data, service_type)
                    if document_data:
                        all_documents.append(document_data)
                        article_info.append({
                            'id': article_data.get('id'),
                            'service_type': service_type,
                            'title': article_data.get('title', 'No title')
                        })
            
            if not all_documents:
                logger.warning("No valid articles found for import")
                return
            
            logger.info(f"Prepared {len(all_documents)} articles for batch import")
            
            # Decide on batch vs individual processing
            if use_batch_api and len(all_documents) >= 10:
                logger.info("Using batch processing with OpenAI Batch API (50% cost savings)")
                await self._import_via_batch_api(all_documents, article_info)
            else:
                logger.info(f"Using individual processing ({len(all_documents)} documents)")
                await self._import_individually(all_documents, article_info)
            
        except Exception as e:
            logger.error(f"Failed to import FAQs: {e}", exc_info=True)
            raise
    
    async def _import_via_batch_api(self, documents: List[Dict[str, Any]], article_info: List[Dict[str, Any]]):
        """Import documents using the batch processing API."""
        try:
            # Create batch job
            async with httpx.AsyncClient(timeout=120.0) as client:
                logger.info("Creating batch processing job...")
                response = await client.post(
                    f"{self.rag_service_url}/api/v1/documents/batch-job",
                    json=documents
                )
                
                if response.status_code == 202:
                    job_info = response.json()
                    job_id = job_info["job_id"]
                    logger.info(f"Batch job created: {job_id}")
                    logger.info(f"Status: {job_info['status']}")
                    logger.info(f"Total documents: {job_info['total_documents']}")
                    logger.info(f"Total chunks: {job_info['total_chunks']}")
                    logger.info(f"Cost savings: {job_info['cost_savings']}")
                    
                    # Monitor job progress
                    if job_info["status"] == "processing":
                        await self._monitor_batch_job(job_id)
                    else:
                        logger.info("Batch job completed immediately (all embeddings were cached)")
                    
                    # Log successful articles
                    for info in article_info:
                        logger.info(
                            f"Successfully imported FAQ article {info['id']} "
                            f"({info['service_type']}) - {info['title']}"
                        )
                    
                elif response.status_code == 400:
                    error_detail = response.json().get('detail', 'Unknown error')
                    logger.warning(f"Batch API not suitable: {error_detail}")
                    logger.info("Falling back to individual processing...")
                    await self._import_individually(documents, article_info)
                else:
                    logger.error(f"Batch job creation failed: {response.status_code} - {response.text}")
                    raise Exception(f"Batch API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Batch import failed: {e}")
            logger.info("Falling back to individual processing...")
            await self._import_individually(documents, article_info)
    
    async def _monitor_batch_job(self, job_id: str):
        """Monitor batch job progress until completion."""
        logger.info(f"Monitoring batch job {job_id}...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            max_attempts = 720  # 1 hour with 5s intervals
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    response = await client.get(
                        f"{self.rag_service_url}/api/v1/documents/batch-job/{job_id}"
                    )
                    
                    if response.status_code == 200:
                        status_info = response.json()
                        status = status_info["status"]
                        
                        if status == "completed":
                            logger.info(f"Batch job {job_id} completed successfully!")
                            progress = status_info.get("progress", {})
                            logger.info(f"Total requests: {progress.get('total_requests', 0)}")
                            logger.info(f"Completed: {progress.get('completed', 0)}")
                            logger.info(f"Failed: {progress.get('failed', 0)}")
                            return
                        elif status in ["failed", "expired", "cancelled"]:
                            logger.error(f"Batch job {job_id} failed with status: {status}")
                            raise Exception(f"Batch job failed: {status}")
                        else:
                            # Still processing
                            progress = status_info.get("progress", {})
                            completed = progress.get('completed', 0)
                            total = progress.get('total_requests', 0)
                            
                            if total > 0:
                                percentage = (completed / total) * 100
                                logger.info(f"Batch job progress: {completed}/{total} ({percentage:.1f}%)")
                            else:
                                logger.info(f"Batch job status: {status}")
                    
                    # Wait before next check
                    await asyncio.sleep(5)
                    attempt += 1
                    
                except Exception as e:
                    logger.warning(f"Error checking batch status: {e}")
                    await asyncio.sleep(5)
                    attempt += 1
            
            logger.warning(f"Batch job monitoring timed out after 1 hour")
    
    async def _import_individually(self, documents: List[Dict[str, Any]], article_info: List[Dict[str, Any]]):
        """Import documents individually (fallback method)."""
        logger.info("Processing articles individually...")
        
        successful = 0
        failed = 0
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i, (document_data, info) in enumerate(zip(documents, article_info)):
                try:
                    response = await client.post(
                        f"{self.rag_service_url}/api/v1/documents/",
                        json=document_data
                    )
                    
                    if response.status_code == 201:
                        result_documents = response.json()
                        logger.info(
                            f"Successfully imported FAQ article {info['id']} "
                            f"({info['service_type']}) - created {len(result_documents)} chunks"
                        )
                        successful += 1
                    else:
                        logger.error(
                            f"Failed to import article {info['id']}: "
                            f"{response.status_code} - {response.text}"
                        )
                        failed += 1
                    
                    # Rate limiting between requests
                    if i < len(documents) - 1:
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Error importing article {info['id']}: {e}")
                    failed += 1
        
        logger.info(f"Individual import completed: {successful} successful, {failed} failed")
    
    async def _fetch_all_articles(self, categories: Dict[str, int]) -> Dict[str, List[Dict]]:
        """Fetch articles from all configured categories."""
        async with aiohttp.ClientSession() as session:
            fetch_tasks = [
                self.zendesk_client.fetch_articles_for_category(
                    session, category_id, service_type
                )
                for service_type, category_id in categories.items()
            ]
            
            results = await asyncio.gather(*fetch_tasks)
            
            # Group results by service_type and filter valid articles
            articles_by_service = {}
            for i, articles in enumerate(results):
                service_type = list(categories.keys())[i]
                valid_articles = [
                    article for article in articles
                    if self.processor.is_valid_article(article)
                ]
                articles_by_service[service_type] = valid_articles
                logger.info(f"Found {len(valid_articles)} valid articles for {service_type}")
            
            return articles_by_service
    
    
    def _prepare_article_for_batch(
        self,
        article: Dict[str, Any], 
        service_type: str
    ) -> Optional[Dict[str, Any]]:
        """Prepare a single FAQ article for batch processing."""
        # Extract and clean content
        title = article.get("title", "")
        body = self.processor.clean_html_content(article.get("body", ""))
        content = f"{title}\n\n{body}".strip()
        
        if not content:
            logger.warning(f"Article {article.get('id')} has no content, skipping")
            return None
        
        # Create metadata
        metadata = self.processor.extract_metadata(article)
        metadata["service_type"] = service_type
        
        # Set company_form based on service_type for compatibility
        if service_type in ['tmi', 'oy', 'ky', 'kyy']:
            metadata["company_form"] = service_type
        else:
            metadata["company_form"] = "general"
        
        return {
            "content": content,
            "metadata": metadata
        }


async def main():
    parser = argparse.ArgumentParser(description="Import FAQ articles from Zendesk")
    parser.add_argument(
        "--category-id", 
        type=int, 
        help="Import from specific category ID only"
    )
    parser.add_argument(
        "--service-type", 
        help="Service type for specific category (required with --category-id)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Fetch and process but don't import to RAG service"
    )
    parser.add_argument(
        "--rag-url", 
        default="http://localhost:8000",
        help="RAG service URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch processing (process articles individually)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.category_id and not args.service_type:
        parser.error("--service-type is required when using --category-id")
    
    # Configuration - these should be set as environment variables
    zendesk_subdomain = os.getenv('ZENDESK_SUBDOMAIN')
    zendesk_email = os.getenv('ZENDESK_EMAIL') 
    zendesk_token = os.getenv('ZENDESK_TOKEN')
    
    if not all([zendesk_subdomain, zendesk_email, zendesk_token]):
        logger.error(
            "Missing Zendesk credentials. Please set ZENDESK_SUBDOMAIN, "
            "ZENDESK_EMAIL, and ZENDESK_TOKEN environment variables."
        )
        sys.exit(1)
    
    # Category mapping - customize this for your Zendesk setup
    if args.category_id:
        categories = {args.service_type: args.category_id}
    else:
        categories = {
            'tmi': 10831661265053,        # Example category IDs
            'oy': 5011347389725,          # Update these for your Zendesk
            'general': 14688742878237,
            'ky': 10831401953821,
            'kyy': 4420200282641
        }
    
    # Initialize services
    zendesk_client = ZendeskClient(zendesk_subdomain, zendesk_email, zendesk_token)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - Articles will be fetched and processed but not imported")
        
        async with aiohttp.ClientSession() as session:
            for service_type, category_id in categories.items():
                articles = await zendesk_client.fetch_articles_for_category(
                    session, category_id, service_type
                )
                processor = ZendeskProcessor()
                valid_articles = [
                    article for article in articles
                    if processor.is_valid_article(article)
                ]
                
                logger.info(f"Service type: {service_type}")
                logger.info(f"  Total articles: {len(articles)}")
                logger.info(f"  Valid articles: {len(valid_articles)}")
                
                for article in valid_articles[:3]:  # Show first 3 as examples
                    title = article.get('title', 'No title')
                    logger.info(f"  - {title}")
    else:
        # Run actual import
        importer = FAQImporter(zendesk_client, args.rag_url)
        use_batch_api = not args.no_batch
        await importer.import_all_faqs(categories, use_batch_api=use_batch_api)


if __name__ == "__main__":
    import os
    asyncio.run(main())