.PHONY: test lint exp-markov-smoke

test:
	PYTHONPATH=src pytest -q

lint:
	ruff check .

exp-markov-smoke:
	PYTHONPATH=src python experiments/smoke.py
