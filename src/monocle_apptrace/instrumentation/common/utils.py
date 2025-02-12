import logging, json
from typing import Callable, Generic, Optional, TypeVar
from threading import local

from opentelemetry.context import attach, detach, get_current, get_value, set_value, Context
from opentelemetry import baggage
from opentelemetry.trace import NonRecordingSpan, Span
from opentelemetry.trace.propagation import _SPAN_KEY
from opentelemetry.sdk.trace import id_generator
from opentelemetry.propagate import inject, extract
from monocle_apptrace.instrumentation.common.constants import MONOCLE_SCOPE_NAME_PREFIX, SCOPE_METHOD_FILE

T = TypeVar('T')
U = TypeVar('U')

logger = logging.getLogger(__name__)

embedding_model_context = {}
token_data = local()
token_data.current_token = None

scope_id_generator = id_generator.RandomIdGenerator()
scopes:dict = {} #TODO: Handle multi-thread/multi-context scopes
http_scopes:list[str] = []

def get_local_token():
    return token_data.current_token

def set_local_token(token):
    token_data.current_token = token

def set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)

def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    # pylint: disable=inconsistent-return-statements
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            logger.warning("Failed to execute %s, error: %s", func.__name__, str(ex))

    return wrapper

def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, handler, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            global global_context
            try:
                # get and log the parent span context if injected by the application
                # This is useful for debugging and tracing of Azure functions
                _parent_span_context = get_current()
                if _parent_span_context is not None and _parent_span_context.get(_SPAN_KEY, None):
                    parent_span: Span = _parent_span_context.get(_SPAN_KEY, None)
                    is_span = isinstance(parent_span, NonRecordingSpan)
                    if is_span:
                        logger.debug(
                            f"Parent span is found with trace id {hex(parent_span.get_span_context().trace_id)}")
            except Exception as e:
                logger.error("Exception in attaching parent context: %s", e)

            val = func(tracer, handler, to_wrap, wrapped, instance, args, kwargs)
            return val

        return wrapper

    return _with_tracer

def set_embedding_model(model_name: str):
    """
    Sets the embedding model in the global context.

    @param model_name: The name of the embedding model to set
    """
    embedding_model_context['embedding_model'] = model_name

def get_embedding_model() -> str:
    """
    Retrieves the embedding model from the global context.

    @return: The name of the embedding model, or 'unknown' if not set
    """
    return embedding_model_context.get('embedding_model', 'unknown')

def set_attribute(key: str, value: str):
    """
    Set a value in the global context for a given key.

    Args:
        key: The key for the context value to set.
        value: The value to set for the given key.
    """
    attach(set_value(key, value))

def get_attribute(key: str) -> str:
    """
    Retrieve a value from the global context for a given key.

    Args:
        key: The key for the context value to retrieve.

    Returns:
        The value associated with the given key.
    """
    return get_value(key)

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_fully_qualified_class_name(instance):
    if instance is None:
        return None
    module_name = instance.__class__.__module__
    qualname = instance.__class__.__qualname__
    return f"{module_name}.{qualname}"

# returns json path like key probe in a dictionary
def get_nested_value(data, keys):
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        elif hasattr(data, key):
            data = getattr(data, key)
        else:
            return None
    return data


def get_keys_as_tuple(dictionary, *keys):
    return tuple(next((value for key, value in dictionary.items() if key.endswith(k) and value is not None), None) for k in keys)

#TODO: handle file/io exceptions
def load_scopes() -> dict:
    methods_data = []
    scope_methods = []
    with open(SCOPE_METHOD_FILE) as f:
        methods_data = json.load(f)
        for method in methods_data:
            if method.get('http_header'):
                http_scopes.append(method.get('http_header'))
            else:
                scope_methods.append(method)

    return scope_methods

def set_scope(scope_name: str, scope_value:str = None) -> None:
    global scope_id_generator
    global scopes
    if scope_value is None:
        scope_value = f"{hex(scope_id_generator.generate_trace_id())}"
    scopes[scope_name] = scope_value
    return

def remove_scope(scope_name: str) -> None:
    global scopes
    scopes.pop(scope_name, None)
    return

def get_scopes() -> dict:
    global scopes
    return scopes.items()

def get_baggage_for_scopes():
    baggage_context:Context = None
    for scope_key, scope_value in get_scopes():
        monocle_scope_name = f"{MONOCLE_SCOPE_NAME_PREFIX}{scope_key}"
        baggage_context = baggage.set_baggage(monocle_scope_name, scope_value, context=baggage_context)
    return baggage_context

def set_scopes_from_baggage(baggage_context:Context):
    for scope_key, scope_value in baggage.get_all(baggage_context):
        if scope_key.startswith(MONOCLE_SCOPE_NAME_PREFIX):
            scope_name = scope_key[len(MONOCLE_SCOPE_NAME_PREFIX):]
            set_scope(scope_name, scope_value)

def extract_http_headers(headers) -> str:
    global http_scopes
    trace_context:Context = extract(headers)
    set_scopes_from_baggage(trace_context)
    #TODO: handle HTTP_TRACEPARENT within extract() with additional mapping
    try:
        traceparent = headers['HTTP_TRACEPARENT']
        traceparent_parts = traceparent.split('-')
        version, trace_id, parent_id, flags = traceparent_parts
    except Exception as e:
        trace_id = ""
    for http_scope in http_scopes:
        if http_scope in headers:
            set_scope(http_scope, headers[http_scope])
        elif f"HTTP_{http_scope.upper().replace("-","_")}" in headers:
            set_scope(http_scope, headers[f"HTTP_{http_scope.upper().replace("-","_")}"])
    return trace_id

def clear_http_scopes():
    global http_scopes
    for http_scope in http_scopes:
        remove_scope(http_scope)

class Option(Generic[T]):
    def __init__(self, value: Optional[T]):
        self.value = value

    def is_some(self) -> bool:
        return self.value is not None

    def is_none(self) -> bool:
        return self.value is None

    def unwrap_or(self, default: T) -> T:
        return self.value if self.is_some() else default

    def map(self, func: Callable[[T], U]) -> 'Option[U]':
        if self.is_some():
            return Option(func(self.value))
        return Option(None)

    def and_then(self, func: Callable[[T], 'Option[U]']) -> 'Option[U]':
        if self.is_some():
            return func(self.value)
        return Option(None)

# Example usage
def try_option(func: Callable[..., T], *args, **kwargs) -> Option[T]:
    try:
        return Option(func(*args, **kwargs))
    except Exception:
        return Option(None)