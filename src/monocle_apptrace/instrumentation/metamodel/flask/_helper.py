from monocle_apptrace.instrumentation.common.instrumentor import start_trace, stop_trace
from monocle_apptrace.instrumentation.common.utils import set_scopes_from_baggage, extract_http_headers, clear_http_scopes
from opentelemetry.propagate import extract
from opentelemetry.context import Context

def flask_pre_processor(args, kwargs):
    token = None
    if args:
        trace_id = extract_http_headers(args[0])
        start_trace(trace_id)

def flask_post_processor(tracer, to_wrap, wrapped, instance, args, kwargs,return_value):
    stop_trace()
    clear_http_scopes()

