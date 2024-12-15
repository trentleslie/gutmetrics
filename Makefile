.PHONY: install format lint type-check test clean all

install:
	poetry install

format:
	poetry run ruff format .

lint:
	poetry run ruff check .

type-check:
	poetry run mypy src/gutmetrics

test:
	poetry run pytest tests/ --cov=gutmetrics --cov-report=term-missing

clean:
	rm -rf .coverage
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf **/__pycache__

check: format lint type-check test

all: clean install check