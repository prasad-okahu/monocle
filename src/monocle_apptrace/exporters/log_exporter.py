from os import linesep
import logging
from typing import Optional, Callable, Sequence, IO
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

class LogSpanExporter(SpanExporter):
    """Implementation of :class:`SpanExporter` that prints spans to the
    logger at INFO level.
    """

    def __init__(
        self,
        service_name: Optional[str] = None,
        formatter: Callable[
            [ReadableSpan], str
        ] = lambda span: span.to_json()
        + linesep,
    ):
        self.formatter = formatter
        self.service_name = service_name
        self.logger = logging.getLogger(__name__)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            self.logger.info(self.formatter(span))
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True