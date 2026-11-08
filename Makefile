.PHONY: tests run evals-help scenarios label evals judge evaluators questions synthetic judge-synthetic explore traces phoenix-upload phoenix-experiment annotate

WEEK4 = cd src/alexey_ai_engineering_buildcamp/week-4
EVALS = uv run python -m pulumi_docs_agent.evals

RESULTS ?= pulumi_docs_agent/evals/scenarios/results/20261107-144942.jsonl
DATA ?= pulumi_docs_agent/evals/synthetic/data/evals_run_2026_11_08_112720_synthetic.json

tests:
	$(WEEK4) && uv run pytest

run:
	$(WEEK4) && uv run pulumi-docs-agent

evals-help:
	$(WEEK4) && $(EVALS) --help

scenarios:
	$(WEEK4) && $(EVALS) scenarios run $(ARGS)

label:
	$(WEEK4) && $(EVALS) scenarios label

evals:
	$(WEEK4) && $(EVALS) scenarios evals $(ARGS)

judge:
	$(WEEK4) && $(EVALS) judge evaluate --data $(RESULTS) $(ARGS)

evaluators:
	$(WEEK4) && $(EVALS) code-evaluators run --data $(RESULTS)

questions:
	$(WEEK4) && $(EVALS) synthetic generate $(ARGS)

synthetic:
	$(WEEK4) && $(EVALS) synthetic run $(ARGS)

judge-synthetic:
	$(WEEK4) && $(EVALS) synthetic judge --data $(DATA) $(ARGS)

explore:
	$(WEEK4) && $(EVALS) synthetic explore

traces:
	$(WEEK4) && $(EVALS) phoenix traces $(ARGS)

phoenix-upload:
	$(WEEK4) && $(EVALS) phoenix upload $(ARGS)

phoenix-experiment:
	$(WEEK4) && $(EVALS) phoenix experiment $(ARGS)

annotate:
	$(WEEK4) && $(EVALS) phoenix annotate $(ARGS)
