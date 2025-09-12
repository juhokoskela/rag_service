.PHONY: help install dev-install build up down logs shell test clean migrate init-db reset-db

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install: ## Install production dependencies
	pip install -r requirements.txt

dev-install: ## Install development dependencies
	pip install -r requirements.txt
	pip install black isort flake8 mypy

# Docker operations
build: ## Build Docker containers
	docker compose build

up: ## Start all services
	docker compose up -d

down: ## Stop all services
	docker compose down

logs: ## Show logs from all services
	docker compose logs -f

shell: ## Open shell in running rag-api container
	docker compose exec rag-api /bin/bash

# Database operations
migrate: ## Run database migrations
	docker compose exec rag-api alembic upgrade head

init-db: ## Initialize database with initial migration
	docker compose exec rag-api alembic upgrade head

reset-db: ## Reset database (WARNING: destroys all data)
	docker compose down -v
	docker compose up -d postgres redis
	sleep 10
	docker compose exec postgres psql -U rag_user -d rag_db -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"
	docker compose up -d rag-api
	sleep 5
	make migrate

# Development
dev: ## Start development environment
	docker compose up --build

# Testing
test: ## Run tests
	docker compose exec rag-api pytest tests/ -v

test-local: ## Run tests locally (outside Docker)
	pytest tests/ -v

# Code quality
format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

lint: ## Run linting checks
	flake8 src/ tests/
	mypy src/

check: format lint ## Format and lint code

# Cleanup
clean: ## Clean up Docker containers and volumes
	docker compose down -v --remove-orphans
	docker system prune -f

clean-cache: ## Clean Python cache files
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete

# Production
prod-up: ## Start production environment
	docker compose -f docker compose.yml -f docker compose.prod.yml up -d

# Health checks
health: ## Check service health
	curl -f http://localhost:8000/health/ || echo "Service not healthy"

health-detailed: ## Check detailed service health
	curl -f http://localhost:8000/health/detailed || echo "Service not healthy"

# Scripts
add-doc: ## Add a document using CLI script (usage: make add-doc CONTENT="text here")
	docker compose exec rag-api python scripts/add_document.py "$(CONTENT)"

add-file: ## Add a file using CLI script (usage: make add-file FILE=path/to/file)
	docker compose exec rag-api python scripts/add_document.py --file "$(FILE)"

batch-add: ## Batch add documents from directory (usage: make batch-add DIR=/path/to/docs)
	docker compose exec rag-api python scripts/batch_add.py "$(DIR)"

import-zendesk: ## Import FAQs from Zendesk (requires ZENDESK_* env vars)
	docker compose exec rag-api python scripts/import_zendesk_faqs.py

import-zendesk-dry: ## Dry run Zendesk FAQ import
	docker compose exec rag-api python scripts/import_zendesk_faqs.py --dry-run

# Database
include .env
export

db-shell: ## Open PostgreSQL shell
	docker compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

db-docs: ## Show document count and recent entries
	docker compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "SELECT COUNT(*) as total_docs FROM documents WHERE deleted_at IS NULL;" -c "SELECT id, LEFT(content, 60) as preview, created_at FROM documents WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT 5;"

db-tables: ## List all database tables
	docker compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "\dt"