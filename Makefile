PYTHON = /usr/local/bin/python3

.PHONY: env bench test tree

env:
	$(PYTHON) check_env.py

bench:
	$(PYTHON) scripts/bench_mps.py

test:
	$(PYTHON) -m pytest -q

tree:
	@find . -maxdepth 3 -not -path '*/.*' | sort
