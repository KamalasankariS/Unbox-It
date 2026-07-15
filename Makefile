PYTHON ?= python

.PHONY: setup data run test api batch clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

## Download the real MIND-small dataset. See README: the original Microsoft blob
## URLs now return HTTP 409 (public access disabled), so set MIND_TRAIN_URL /
## MIND_DEV_URL to a mirror you have access to, or place the extracted TSVs at
## data/MINDsmall_train/ and data/MINDsmall_dev/ by hand.
data:
	$(PYTHON) -m newspush.data.download

run:
	$(PYTHON) -m newspush.pipeline

test:
	$(PYTHON) -m pytest -q

api:
	$(PYTHON) -m uvicorn newspush.serving.api:app --reload --port 8000

batch:
	$(PYTHON) -m newspush.serving.batch

clean:
	rm -rf runs/* .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
