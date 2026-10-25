.PHONY: tests run

tests:
	cd src/alexey_ai_engineering_buildcamp/week-4 && uv run pytest

run:
	cd src/alexey_ai_engineering_buildcamp/week-4 && uv run pulumi-docs-agent
