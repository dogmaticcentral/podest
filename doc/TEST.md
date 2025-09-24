# Testing

Install test ordering package (with .venv activated)
```bash
pip install pytest-ordering
```

```bash
python -m pytest tests/
```
or to stop after first failure

```bash
python -m pytest -x tests/
```

One line per test, inclusive of passed tests:
```bash
python -m pytest -vv -x tests/
```

