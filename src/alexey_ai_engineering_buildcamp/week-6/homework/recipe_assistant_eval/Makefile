.PHONY: run evals-help scenarios label csv judge judge-eval synthetic-generate synthetic-run synthetic-judge synthetic-report

EVALS = uv run python -m recipe_assistant.evals

run:
	uv run recipe-assistant

evals-help:
	$(EVALS) --help

scenarios:
	$(EVALS) scenarios run $(ARGS)

label:
	$(EVALS) scenarios label

csv:
	$(EVALS) scenarios csv

judge:
	$(EVALS) judge run $(ARGS)

judge-eval:
	$(EVALS) judge evaluate $(ARGS)

SYN = recipe_assistant/evals/synthetic

synthetic-generate:
	$(EVALS) synthetic generate $(ARGS)

synthetic-run:
	$(EVALS) scenarios run --scenarios $(SYN)/synthetic_scenarios.csv --output $(SYN)/results/results.json

synthetic-judge:
	$(EVALS) judge run --results $(SYN)/results/results.json --output $(SYN)/results/results_judged.json

synthetic-report:
	$(EVALS) synthetic report $(ARGS)
