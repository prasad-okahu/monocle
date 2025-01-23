from monocle_apptrace.instrumentation.common.instrumentor import start_trace, stop_trace
def flask_pre_processor(args, kwargs):
    token = None
    if args and 'HTTP_TRACEPARENT' in args[0]:
        traceparent = args[0]['HTTP_TRACEPARENT']
        traceparent_parts = traceparent.split('-')
        version, trace_id, parent_id, flags = traceparent_parts
        start_trace(trace_id)

def flask_post_processor(tracer, to_wrap, wrapped, instance, args, kwargs,return_value):
    stop_trace()

