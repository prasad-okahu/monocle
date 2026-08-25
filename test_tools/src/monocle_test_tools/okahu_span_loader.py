import logging
import os
from typing import Any, Dict, List, Optional
import requests
from opentelemetry.sdk.trace import ReadableSpan
from monocle_test_tools.file_span_loader import JSONSpanLoader

logger = logging.getLogger(__name__)


class OkahuSpanLoader:
    """Utility class to load spans from Okahu trace service.

    Uses the Okahu REST API:
        - GET /api/v1/apps/<app_name>/traces?duration_fact=<fact>&fact_ids=<id>
          Get traces matching a fact (e.g. ``agent_sessions``).
        - GET /api/v1/apps/<app_name>/traces/<trace_id>/spans
          Get spans for a trace, optionally filtered by session.

    Base URL defaults to https://api.okahu.co and can be overridden
    with the OKAHU_API_ENDPOINT environment variable.
    """

    # Constants
    AGENT_SESSIONS_SCOPE = "agent_sessions"
    OKAHU_BASE_URL = "https://api.okahu.co"

    RESOURCE_NAMESPACES = ("apps", "workflows")

    @staticmethod
    def _get_api_base(endpoint: Optional[str] = None) -> str:
        """Return the Okahu API base URL (no trailing slash).

        ``or OKAHU_BASE_URL`` rather than a getenv default: the variable is
        set-but-empty under pytest (tests/integration/__init__.py setdefaults it
        to ""), and an empty base builds a hostless URL that fails with
        MissingSchema instead of falling back to prod.
        """
        return (endpoint or os.environ.get("OKAHU_API_ENDPOINT")
                or OkahuSpanLoader.OKAHU_BASE_URL).rstrip("/")

    @staticmethod
    def _get_headers(api_key: Optional[str] = None) -> dict:
        """Return common request headers."""
        key = api_key or os.environ.get("OKAHU_API_KEY")
        if not key:
            raise ValueError("OKAHU_API_KEY is not configured. Set the environment variable or pass api_key.")
        return {
            "Content-Type": "application/json",
            "x-api-key": key
        }

    @staticmethod
    def _get_resource(base: str, path_suffix: str, headers: dict,
                      params: Optional[dict] = None, timeout: int = 30,
                      context_msg: str = "") -> Any:
        """GET an application-scoped resource, trying each known namespace.

        ``path_suffix`` is everything after the application name, e.g.
        ``"traces"`` or ``"traces/<id>/spans"``. Only a 404 moves on to the next
        namespace; any other error is the caller's to see.
        """
        last_error: Optional[requests.HTTPError] = None
        for namespace in OkahuSpanLoader.RESOURCE_NAMESPACES:
            url = f"{base}/api/v1/{namespace}/{path_suffix}"
            try:
                return OkahuSpanLoader._do_get(
                    url, headers, params=params, timeout=timeout, context_msg=context_msg
                )
            except requests.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status != 404:
                    raise
                logger.debug("Okahu %r namespace returned 404 for %s", namespace, path_suffix)
                last_error = exc
        raise last_error

    @staticmethod
    def _do_get(url: str, headers: dict, params: Optional[dict] = None,
                timeout: int = 30, context_msg: str = "") -> Any:
        """Execute a GET request with standard error handling."""
        try:
            response = requests.get(url=url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise ConnectionError(f"Okahu request timed out ({context_msg}): {exc}") from exc
        except requests.HTTPError as exc:
            raise
        except requests.RequestException as exc:
            raise ConnectionError(f"Failed to reach Okahu service ({context_msg}): {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise ConnectionError(
                f"Okahu returned invalid JSON ({context_msg}): {response.text}"
            ) from exc

    @staticmethod
    def _window_params(start_time: Optional[str], end_time: Optional[str]) -> dict:
        """The time-window query params, omitting whichever end was not given.

        Returned as a dict to merge so that a call with no window leaves `params`
        exactly as it was -- callers rely on an empty params becoming None.
        """
        window = {}
        if start_time is not None:
            window["start_time"] = start_time
        if end_time is not None:
            window["end_time"] = end_time
        return window

    @staticmethod
    def _unwrap_list(data: Any, wrapper_keys: tuple, context_msg: str = "") -> list:
        """Unwrap a list from a possible dict wrapper."""
        if isinstance(data, dict):
            for key in wrapper_keys:
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise ConnectionError(
                f"Okahu response is a dict but no known list key found ({context_msg}). "
                f"Keys: {list(data.keys())}"
            )
        if isinstance(data, list):
            return data
        raise ConnectionError(
            f"Expected a list from Okahu ({context_msg}), got: {type(data).__name__}"
        )

    # ------------------------------------------------------------------ #
    #  Public helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_fact_ids(
        workflow_name: str,
        fact_name: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[str]:
        """Fetch the ids of every fact of one level in a workflow.

        Uses:  GET /api/v1/workflows/<wf>/facts/<fact_name>/ids
               ?duration_fact=<fact_name>&breakdown_filter=<fact_name>

        This is the entry point for any fact level above a trace -- agent
        requests, sessions, conversations. A trace-level set comes from
        ``get_trace_ids`` instead.

        The response keys ``fact_ids`` to an object keyed by id, not to a list,
        so the ids are its keys and the order is the server's. Each value holds
        that fact's timing and status, and *sometimes* a ``traces`` array -- only
        for the first entry, in practice. That array is deliberately ignored:
        relying on it would make one fact behave differently from the rest, so
        every fact's traces are fetched uniformly with ``get_trace_ids``.

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            fact_name: The Okahu fact level (e.g. ``agent_requests``), already
                mapped -- this goes straight into the URL path.
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds.
            start_time: Optional window start.
            end_time: Optional window end.

        Returns:
            The fact ids, in the order the server returned them.
        """
        base = OkahuSpanLoader._get_api_base(endpoint)
        headers = OkahuSpanLoader._get_headers(api_key)
        url = f"{base}/api/v1/workflows/{workflow_name}/facts/{fact_name}/ids"
        params = {"duration_fact": fact_name, "breakdown_filter": fact_name}
        params.update(OkahuSpanLoader._window_params(start_time, end_time))

        data = OkahuSpanLoader._do_get(
            url, headers, params=params, timeout=timeout,
            context_msg=f"{fact_name} ids in workflow '{workflow_name}'")

        fact_ids = (data or {}).get("fact_ids")
        if isinstance(fact_ids, dict):
            return list(fact_ids)
        if isinstance(fact_ids, list):
            return [item.get("fact_id") if isinstance(item, dict) else item
                    for item in fact_ids]
        if fact_ids is None:
            return []
        raise ConnectionError(
            f"Okahu returned an unexpected 'fact_ids' for {fact_name} in workflow "
            f"'{workflow_name}': {type(fact_ids).__name__}")

    @staticmethod
    def get_trace_ids(
        workflow_name: str,
        fact_name: Optional[str] = None,
        fact_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[str]:
        """Fetch trace IDs from Okahu filtered by a fact.

        Uses:  GET /api/v1/workflows/<wf>/traces?duration_fact=<fact_name>&fact_ids=<fact_id>

        With no fact filter the query returns every trace in the workflow (or in
        the time window, when one is given) -- that is how a test-case set is
        enumerated. Pass both fact_name and fact_id to narrow to one fact.

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            fact_name: The fact to filter by (e.g. ``agentic_session``). Optional,
                but only together with fact_id.
            fact_id: The fact value (e.g. a session ID). Optional, but only
                together with fact_name.
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds.

        Returns:
            A list of trace ID strings.

        Raises:
            ValueError: If exactly one of fact_name / fact_id is given. Half a
                filter is a mistake, not a mode.
        """
        if (fact_name is None) != (fact_id is None):
            raise ValueError(
                "fact_name and fact_id must be given together or not at all; "
                f"got fact_name={fact_name!r}, fact_id={fact_id!r}")

        base = OkahuSpanLoader._get_api_base(endpoint)
        headers = OkahuSpanLoader._get_headers(api_key)
        params = {}
        if fact_name is not None:
            params["duration_fact"] = fact_name
            params["fact_ids"] = fact_id
        params.update(OkahuSpanLoader._window_params(start_time, end_time))

        data = OkahuSpanLoader._get_resource(
            base, f"{workflow_name}/traces", headers, params=params, timeout=timeout,
            context_msg=(f"traces for {fact_name}='{fact_id}' in workflow '{workflow_name}'"
                         if fact_name else f"traces in workflow '{workflow_name}'")
        )

        trace_list = OkahuSpanLoader._unwrap_list(
            data, ("traces", "data", "results"),
            context_msg=f"traces for {fact_name}='{fact_id}'"
        )

        trace_ids = []
        for item in trace_list:
            if isinstance(item, dict) and "trace_id" in item:
                trace_ids.append(item["trace_id"])
            elif isinstance(item, str):
                trace_ids.append(item)

        logger.debug(
            "Found %d trace(s) for %s='%s' in workflow '%s'",
            len(trace_ids), fact_name, fact_id, workflow_name,
        )
        return trace_ids

    @staticmethod
    def get_spans(
        workflow_name: str,
        trace_id: str,
        filter_fact: Optional[str] = None,
        filter_fact_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[ReadableSpan]:
        """Fetch spans from Okahu for a given trace_id.

        Uses:  GET /api/v1/workflows/<wf>/traces/<trace_id>/spans
        Optionally appends ``?filter_fact=<fact>&filter_fact_id=<id>``
        to filter spans server-side (e.g. by session).

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            trace_id: The trace ID (hex string) to fetch spans for.
            filter_fact: Optional server-side span filter fact name.
            filter_fact_id: Optional server-side span filter fact value.
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds.

        Returns:
            A list of ReadableSpan instances.

        Raises:
            ValueError: If OKAHU_API_KEY is not configured.
            ConnectionError: If the request to Okahu fails.
        """
        # Strip 0x prefix if present
        trace_id = trace_id.replace("0x", "")

        base = OkahuSpanLoader._get_api_base(endpoint)
        headers = OkahuSpanLoader._get_headers(api_key)
        params = {}
        if filter_fact and filter_fact_id:
            params["filter_fact"] = filter_fact
            params["filter_fact_id"] = filter_fact_id
        params.update(OkahuSpanLoader._window_params(start_time, end_time))

        span_data_list = OkahuSpanLoader._get_resource(
            base, f"{workflow_name}/traces/{trace_id}/spans", headers,
            params=params or None, timeout=timeout,
            context_msg=f"spans for trace_id '{trace_id}' in workflow '{workflow_name}'"
        )

        span_data_list = OkahuSpanLoader._unwrap_list(
            span_data_list, ("spans", "batch", "data", "results", "trace_spans"),
            context_msg=f"spans for trace_id '{trace_id}'"
        )

        span_list = []
        for item in span_data_list:
            span = JSONSpanLoader._from_dict(span_data=item)
            span_list.append(span)
        # verify that there's a span with span.attributes["span.type"] == "workflow" otherwise raise HttpError 404
        if not any(span.attributes.get("span.type") == "workflow" for span in span_list):
            raise requests.HTTPError(f"No workflow span found in trace '{trace_id}' - possible invalid trace ID or trace not fully ingested yet.")

        logger.debug("Loaded %d spans from Okahu for trace_id '%s'", len(span_list), trace_id)
        return span_list

    @staticmethod
    def load_by_session(
        workflow_name: str,
        session_id: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> List[ReadableSpan]:
        """Fetch all spans for every trace in a session.

        This is a convenience wrapper around ``load_by_scope()`` that uses
        the standard "agent_sessions" scope name.

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            session_id: The agent session ID.
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds.

        Returns:
            A flat list of ReadableSpan instances from all matching traces.

        Raises:
            ConnectionError: If no traces found or API call fails.
        """
        return OkahuSpanLoader.load_by_scope(
            workflow_name=workflow_name,
            scope_name=OkahuSpanLoader.AGENT_SESSIONS_SCOPE,
            scope_id=session_id,
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
        )

    @staticmethod
    def load_by_scope(
        workflow_name: str,
        scope_name: str,
        scope_id: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[ReadableSpan]:
        """Fetch all spans for every trace matching a custom scope.

        This is a generic method that works with any Okahu fact/scope.
        For example:
        - scope_name="agent_sessions", scope_id="session_123"
        - scope_name="test_id", scope_id="test_456"
        - scope_name="my_custom_scope", scope_id="custom_789"

        1. GET traces with ``duration_fact=<scope_name>&fact_ids=<scope_id>``
        2. For each trace, GET spans with ``filter_fact=<scope_name>&filter_fact_id=<scope_id>``
        3. Return ReadableSpan objects.

        Args:
            workflow_name: The workflow / service name registered in Okahu.
            scope_name: The name of the scope/fact to filter by.
            scope_id: The scope/fact value (e.g., session ID, test ID, etc.).
            endpoint: Okahu API base URL override.
            api_key: Okahu API key override.
            timeout: Request timeout in seconds.

        Returns:
            A flat list of ReadableSpan instances from all matching traces.

        Raises:
            ValueError: If scope_name or scope_id is empty.
            ConnectionError: If no traces found or API call fails.
        """
        # Validate inputs
        if not scope_name or not scope_name.strip():
            raise ValueError("scope_name cannot be empty")
        if not scope_id or not scope_id.strip():
            raise ValueError("scope_id cannot be empty")

        trace_ids = OkahuSpanLoader.get_trace_ids(
            workflow_name,
            fact_name=scope_name,
            fact_id=scope_id,
            endpoint=endpoint, api_key=api_key, timeout=timeout,
            start_time=start_time, end_time=end_time,
        )
        if not trace_ids:
            raise ConnectionError(
                f"No traces found for {scope_name}='{scope_id}' in workflow '{workflow_name}'"
            )

        all_spans: List[ReadableSpan] = []
        for tid in trace_ids:
            spans = OkahuSpanLoader.get_spans(
                workflow_name, tid,
                filter_fact=scope_name,
                filter_fact_id=scope_id,
                endpoint=endpoint, api_key=api_key, timeout=timeout,
                start_time=start_time, end_time=end_time,
            )
            all_spans.extend(spans)

        logger.debug(
            "Loaded %d total spans across %d trace(s) for %s='%s'",
            len(all_spans), len(trace_ids), scope_name, scope_id,
        )
        return all_spans
