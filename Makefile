.PHONY: test reproduce quick
test:
	python -m pytest -q
reproduce:
	python experiments/run_all.py
quick:
	python experiments/run_all.py --quick
