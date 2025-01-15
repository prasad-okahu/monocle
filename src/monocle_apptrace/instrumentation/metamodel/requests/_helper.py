from opentelemetry.propagate import inject
def request_pre_processor(args, kwargs):
    # add traceparent to the request headers in kwargs
    if 'headers' not in kwargs:
        headers = {}
    else:
        headers = kwargs['headers']
    inject(headers)
    kwargs['headers'] = headers

