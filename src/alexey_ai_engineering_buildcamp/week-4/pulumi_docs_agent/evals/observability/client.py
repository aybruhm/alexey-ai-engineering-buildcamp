import os

import dotenv
from phoenix.client import Client

from ..common.paths import WEEK_DIR

DEFAULT_PROJECT = "pulumi-docs-agent"
ANNOTATION_IDENTIFIER = "evals-v1"


def get_client() -> Client:
    dotenv.load_dotenv(WEEK_DIR / ".env")
    return Client(
        base_url=os.environ["PHOENIX_COLLECTOR_ENDPOINT"],
        api_key=os.environ["PHOENIX_API_KEY"],
    )


def project_name() -> str:
    dotenv.load_dotenv(WEEK_DIR / ".env")
    return os.getenv("PHOENIX_PROJECT_NAME", DEFAULT_PROJECT)
