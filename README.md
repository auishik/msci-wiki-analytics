# Wikipedia Word-Frequency Dictionary

A FastAPI server that generates word-frequency dictionaries by traversing Wikipedia articles up to a specified depth.

## Installation

Requires Python 3.12+ and [Poetry](https://python-poetry.org/).

```bash
poetry install
```

## Running the Server

```bash
poetry run uvicorn src.main:app --reload
```

The server starts at `http://localhost:8000`.

## API Endpoints

### GET /word-frequency

Generate word frequencies for a Wikipedia article and its linked articles.

**Parameters:**
- `article` (string) - Wikipedia article title
- `depth` (int) - Traversal depth (0 = only the article, 1 = article + linked articles, etc.)

**Example:**
```bash
curl "http://localhost:8000/word-frequency?article=Python_(programming_language)&depth=0"
```

### POST /keywords

Generate filtered word frequencies, excluding common words.

**Request Body:**
```json
{
  "article": "Python_(programming_language)",
  "depth": 0,
  "ignore_list": ["the", "a", "is", "of"],
  "percentile": 50
}
```

- `ignore_list` - Words to exclude from results
- `percentile` - Filter to top N percentile of words by frequency

**Example:**
```bash
curl -X POST "http://localhost:8000/keywords" \
  -H "Content-Type: application/json" \
  -d '{"article": "Python_(programming_language)", "depth": 0, "ignore_list": ["the", "a"], "percentile": 90}'
```

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run a specific test file
poetry run pytest tests/test_wikipedia.py
```

## Linting

```bash
poetry run pre-commit run --all-files
```
