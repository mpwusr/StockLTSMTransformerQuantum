.PHONY: test test-verbose test-cov test-quick lint clean install-test help

# Default target
help:
	@echo "Available targets:"
	@echo "  make test          - Run all tests"
	@echo "  make test-verbose  - Run all tests with verbose output"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make test-quick    - Run tests (stop on first failure)"
	@echo "  make lint          - Run linting checks"
	@echo "  make clean         - Remove cache and build artifacts"
	@echo "  make install-test  - Install test dependencies"

# Install test dependencies
install-test:
	pip install -r requirements-test.txt

# Run all tests
test:
	python3 -m pytest tests/ -v --tb=short

# Run all tests with verbose output
test-verbose:
	python3 -m pytest tests/ -v --tb=long -s

# Run tests with coverage report
test-cov:
	python3 -m pytest tests/ -v --tb=short --cov=. --cov-report=term-missing --cov-report=html

# Run tests, stop on first failure
test-quick:
	python3 -m pytest tests/ -x --tb=short

# Lint checks
lint:
	python3 -m py_compile config.py
	python3 -m py_compile models.py
	python3 -m py_compile main.py

# Clean up cache and artifacts
clean:
	rm -rf __pycache__ tests/__pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
