from monocle_apptrace.instrumentation.common.instrumentor import propagate_trace_id
def flask_pre_processor(args, kwargs):
    if args and 'HTTP_TRACEPARENT' in args[0]:
        traceparent = args[0]['HTTP_TRACEPARENT']
        traceparent_parts = traceparent.split('-')
        version, trace_id, parent_id, flags = traceparent_parts
        propagate_trace_id(trace_id)


