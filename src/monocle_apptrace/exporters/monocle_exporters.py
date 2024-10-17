from typing import Dict
import os
from opentelemetry.sdk.trace.export import SpanExporter
from monocle_apptrace.exporters.aws.s3_exporter import S3SpanExporter
from monocle_apptrace.exporters.azure.blob_exporter import AzureBlobSpanExporter
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.exporters.log_exporter import LogSpanExporter
from monocle_apptrace.exporters.okahu.okahu_exporter import OkahuSpanExporter

monocle_exporters:Dict[str, SpanExporter.__class__] = [
    {"s3": S3SpanExporter},
    {"blob": AzureBlobSpanExporter},
    {"okahu": OkahuSpanExporter},
    {"file": FileSpanExporter},
    {"log": LogSpanExporter}
]

def get_monocle_exporter() -> SpanExporter:
    exporter_name = os.environ.get("MONOCLE_EXPORTER", "file")
    exporter_class = monocle_exporters.get(exporter_name, FileSpanExporter)
    try:
        return exporter_class()
    except Exception as ex:
        print(ex)
        return LogSpanExporter()