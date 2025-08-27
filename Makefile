.PHONY: help install install-dev test test-cov lint format clean run train

help:  ## Show this help message
	@echo "Personal News Aggregator - Development Commands"
	@echo "=============================================="
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

test:  ## Run tests
	python -m pytest tests/ -v

test-cov:  ## Run tests with coverage
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:  ## Run linting checks
	flake8 src/ tests/ main.py config.py
	mypy src/ main.py config.py
	isort --check-only src/ tests/ main.py config.py

format:  ## Format code
	black src/ tests/ main.py config.py
	isort src/ tests/ main.py config.py

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

run:  ## Run the news aggregator
	python main.py

train:  ## Train the model
	python train_model.py

setup-env:  ## Setup development environment
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  # On Unix/macOS"
	@echo "  venv\\Scripts\\activate     # On Windows"

pre-commit:  ## Install pre-commit hooks
	pre-commit install

check-all: lint test  ## Run all checks (lint + test)

build:  ## Build the package
	python setup.py sdist bdist_wheel

install-package:  ## Install the package in development mode
	pip install -e .
