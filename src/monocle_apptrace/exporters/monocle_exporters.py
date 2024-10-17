from typing import Dict
import os
from opentelemetry.sdk.trace.export import SpanExporter
import exporters
import exporters.aws
import exporters.aws.s3_exporter
import exporters.azure
import exporters.azure.blob_exporter
import exporters.file_exporter
import exporters.log_exporter
import exporters.okahu
import exporters.okahu.okahu_exporter

monocle_exporters:Dict[str, SpanExporter.__class__] = [
    {"s3": exporters.aws.s3_exporter.S3SpanExporter},
    {"blob": exporters.azure.blob_exporter.AzureBlobSpanExporter},
    {"okahu": exporters.okahu.okahu_exporter.OkahuSpanExporter},
    {"file": exporters.file_exporter.FileSpanExporter},
    {"log": exporters.log_exporter.LogSpanExporter}
]

def get_monocle_exporter() -> SpanExporter:
    exporter_name = os.environ.get("MONOCLE_EXPORTER", "file")
    exporter_class = monocle_exporters.get(exporter_name, exporters.file_exporter.FileSpanExporter)
    try:
        return exporter_class()
    except Exception as ex:
        print(ex)
        return exporters.log_exporter.LogSpanExporter()