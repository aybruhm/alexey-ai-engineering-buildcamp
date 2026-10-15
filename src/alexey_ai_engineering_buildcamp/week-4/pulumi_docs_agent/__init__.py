import os
from pathlib import Path

import dotenv
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from openinference.semconv.resource import ResourceAttributes
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

WRK_DIR = Path(__file__).parent
dotenv.load_dotenv(str(WRK_DIR.parent / ".env"), override=True)

# Set up the tracer provider
tracer_provider = TracerProvider(
    resource=Resource.create({
        ResourceAttributes.PROJECT_NAME: os.getenv(
            "PHOENIX_PROJECT_NAME",
            "default",
        )
    })
)
trace.set_tracer_provider(tracer_provider)

# Add the OpenInference span processor
endpoint = f"{os.environ['PHOENIX_COLLECTOR_ENDPOINT']}/v1/traces"

# If you are using a local instance without auth, ignore these headers
headers = {
    "Authorization": f"Bearer {os.environ['PHOENIX_API_KEY']}",
    "X-Project-Name": os.getenv(
        "PHOENIX_PROJECT_NAME",
        "default",
    ),
}
exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)

tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
