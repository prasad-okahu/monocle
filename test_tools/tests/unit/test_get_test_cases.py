"""Building FluentTestCases from the Okahu /evals/report discovery endpoint.

Discovery mode is selected by the *absence* of fact_ids: the server enumerates
every fact in the window and returns one row per (fact_id, eval_name). Those rows
are grouped back into one test case per fact so the generated cases feed straight
into the testcase= plumbing on with_trace_source / run_agent / check_eval.
"""
import pytest

from monocle_test_tools.evals.okahu_eval import OkahuEval
from monocle_test_tools.fluent_api import get_test_cases
from monocle_test_tools.schema import FactID
from monocle_test_tools.testcase import Eval, FluentTestCase


@pytest.fixture(autouse=True)
def _okahu_env(monkeypatch):
    monkeypatch.setenv("OKAHU_API_KEY", "test-key")
    monkeypatch.setenv("OKAHU_API_ENDPOINT", "https://api.example")


def _row(fact_id, eval_name, label=None, latest_label=None, eval_found=True):
    """One /evals/report result row, per (fact_id, eval_name).

    ``authoritative`` and each ``latest`` entry are run envelopes; the label sits
    inside their ``eval_result``. See TestLiveResponseShape for the captured shape.
    """
    row = {"fact_id": fact_id, "eval_name": eval_name, "eval_found": eval_found}
    if label is not None:
        row["authoritative"] = {
            "eval_result": {"label": label, "explanation": "because"},
            "eval_timestamp": "2026-08-13T17:01:37.646594Z", "category": "llm"}
    if latest_label is not None:
        row["latest"] = [{
            "eval_result": {"label": latest_label, "explanation": "newest"},
            "eval_timestamp": "2026-08-13T17:01:37.646594Z", "category": "llm"}]
    return row


@pytest.fixture(name="post")
def post_fixture(monkeypatch):
    """Stub requests.post, recording calls and replaying queued pages."""
    calls = []
    pages = []

    class _Response:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200
            self.text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "body": json})
        return _Response(pages[len(calls) - 1] if len(calls) <= len(pages) else {"results": []})

    monkeypatch.setattr("monocle_test_tools.evals.okahu_eval.requests.post", fake_post)
    return {"calls": calls, "pages": pages}


def _queue(post, *payloads):
    post["pages"].extend(payloads)


WINDOW = {"workflow_name": "wf", "start_time": "2026-05-01", "end_time": "2026-06-30"}


class TestRequestBody:

    def test_posts_to_the_report_path(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW)

        assert post["calls"][0]["url"] == (
            "https://api.example/api/v1/workflows/wf/evals/report")

    def test_sends_the_api_key(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW)

        assert post["calls"][0]["headers"]["x-api-key"] == "test-key"

    def test_body_carries_the_required_window(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW)

        body = post["calls"][0]["body"]
        assert body["fact_name"] == "traces"
        assert body["start_time"] == "2026-05-01"
        assert body["end_time"] == "2026-06-30"

    def test_omits_fact_ids_to_select_discovery_mode(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW)

        assert "fact_ids" not in post["calls"][0]["body"]

    def test_category_defaults_to_llm_as_a_list(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW)

        assert post["calls"][0]["body"]["category"] == ["llm"]

    def test_category_string_is_wrapped(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, category="manual")

        assert post["calls"][0]["body"]["category"] == ["manual"]

    def test_category_list_passes_through(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, category=["llm", "manual"])

        assert post["calls"][0]["body"]["category"] == ["llm", "manual"]

    def test_fact_name_is_mapped_to_the_okahu_name(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, fact_name="agentic_turns")

        assert post["calls"][0]["body"]["fact_name"] == "agent_requests"

    def test_eval_names_omitted_when_no_eval_name(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW)

        assert "eval_names" not in post["calls"][0]["body"]

    def test_eval_name_becomes_a_one_item_eval_names(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name="hallucination")

        assert post["calls"][0]["body"]["eval_names"] == ["hallucination"]

    def test_custom_eval_name_is_rejected(self, post):
        with pytest.raises(ValueError, match="custom"):
            OkahuEval.get_test_cases(**WINDOW, eval_name="custom")

    def test_page_size_is_sent(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, page_size=250)

        assert post["calls"][0]["body"]["page_size"] == 250


class TestPagination:

    def test_follows_the_next_page_token(self, post):
        _queue(post,
               {"results": [_row("aaa", "hallucination", "minor")],
                "next_page_token": "tok-2"},
               {"results": [_row("bbb", "hallucination", "major")]})

        cases = OkahuEval.get_test_cases(**WINDOW)

        assert len(post["calls"]) == 2
        assert post["calls"][1]["body"]["page_token"] == "tok-2"
        assert [c.name for c in cases] == ["aaa", "bbb"]

    def test_stops_when_no_next_page_token(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor")]})

        OkahuEval.get_test_cases(**WINDOW)

        assert len(post["calls"]) == 1

    def test_empty_next_page_token_stops(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor")],
                      "next_page_token": ""})

        OkahuEval.get_test_cases(**WINDOW)

        assert len(post["calls"]) == 1


class TestMapping:

    def test_one_case_per_fact_with_a_factid_input(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor_hallucination")]})

        cases = OkahuEval.get_test_cases(**WINDOW)

        assert cases == [FluentTestCase(
            name="aaa",
            input=FactID(fact_id="aaa", fact_name="traces", source="okahu"),
            evals=[Eval(name="hallucination", result="minor_hallucination")])]

    def test_several_evals_group_onto_one_fact(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor"),
                                  _row("aaa", "sentiment", "positive")]})

        cases = OkahuEval.get_test_cases(**WINDOW)

        assert len(cases) == 1
        assert cases[0].evals == [Eval(name="hallucination", result="minor"),
                                  Eval(name="sentiment", result="positive")]

    def test_distinct_facts_become_distinct_cases(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor"),
                                  _row("bbb", "hallucination", "major")]})

        cases = OkahuEval.get_test_cases(**WINDOW)

        assert [c.name for c in cases] == ["aaa", "bbb"]

    def test_fact_id_is_normalized_to_bare_hex(self, post):
        _queue(post, {"results": [_row("0xaaa", "hallucination", "minor")]})

        cases = OkahuEval.get_test_cases(**WINDOW)

        assert cases[0].input.fact_id == "aaa"

    def test_user_facing_fact_name_is_kept_on_the_factid(self, post):
        """The body is sent the mapped name; the FactID keeps the caller's."""
        _queue(post, {"results": [_row("aaa", "hallucination", "minor")]})

        cases = OkahuEval.get_test_cases(**WINDOW, fact_name="agentic_turns")

        assert cases[0].input.fact_name == "agentic_turns"

    def test_falls_back_to_latest_when_no_authoritative(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", latest_label="major")]})

        cases = OkahuEval.get_test_cases(**WINDOW)

        assert cases[0].evals == [Eval(name="hallucination", result="major")]

    def test_authoritative_wins_over_latest(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor",
                                       latest_label="major")]})

        cases = OkahuEval.get_test_cases(**WINDOW)

        assert cases[0].evals == [Eval(name="hallucination", result="minor")]

    def test_unlabeled_row_is_skipped(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor"),
                                  _row("aaa", "sentiment")]})

        cases = OkahuEval.get_test_cases(**WINDOW)

        assert cases[0].evals == [Eval(name="hallucination", result="minor")]

    def test_fact_with_no_labels_produces_no_case(self, post):
        """An empty evals list would raise in check_eval -- emit nothing instead."""
        _queue(post, {"results": [_row("aaa", "hallucination")]})

        assert OkahuEval.get_test_cases(**WINDOW) == []

    def test_eval_not_found_row_is_skipped(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor",
                                       eval_found=False)]})

        assert OkahuEval.get_test_cases(**WINDOW) == []

    def test_no_results_gives_no_cases(self, post):
        _queue(post, {"results": []})

        assert OkahuEval.get_test_cases(**WINDOW) == []


class TestFluentEntryPoint:

    def test_delegates_to_okahu(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor")]})

        cases = get_test_cases(source="okahu", **WINDOW)

        assert [c.name for c in cases] == ["aaa"]

    def test_eval_name_is_forwarded(self, post):
        _queue(post, {"results": []})

        get_test_cases(source="okahu", eval_name="hallucination", **WINDOW)

        assert post["calls"][0]["body"]["eval_names"] == ["hallucination"]

    def test_defaults_to_okahu(self, post):
        _queue(post, {"results": []})

        get_test_cases(**WINDOW)

        assert post["calls"][0]["url"].endswith("/evals/report")

    @pytest.mark.parametrize("source", ["file", "local", "s3"])
    def test_other_sources_are_rejected(self, source):
        with pytest.raises(ValueError, match="only 'okahu'"):
            get_test_cases(source=source, **WINDOW)


def test_exported_from_the_package():
    import monocle_test_tools

    assert monocle_test_tools.get_test_cases is get_test_cases


class TestLiveResponseShape:
    """Pinned against a real /evals/report response captured 2026-08-24.

    The label lives at authoritative.eval_result.label -- one level deeper than
    the row. An earlier version read authoritative.label, which silently matched
    nothing and returned zero test cases for every real response.
    """

    LIVE_RESPONSE = {
        "app_name": "adk-travel-agent_vtykgu",
        "eval_names": ["user_input_validity"],
        "fact_name": "traces",
        "category": ["llm", "manual"],
        "results": [
            {
                "fact_id": "9ade6084ba144b138090d64d1a082450",
                "eval_id": "custom_evaluation__generic__user_input_validity",
                "eval_name": "user_input_validity",
                "eval_found": True,
                "authoritative": {
                    "eval_result": {
                        "template_name": "user_input_validity",
                        "label": "valid",
                        "explanation": "The user's input is a request to book a flight...",
                        "category": "manual",
                    },
                    "eval_timestamp": "2026-08-13T17:01:37.646594Z",
                    "category": "manual",
                    "finish_type": "success",
                },
                "latest": [
                    {
                        "eval_result": {"template_name": "user_input_validity",
                                        "label": "valid", "category": "manual"},
                        "eval_timestamp": "2026-08-13T17:01:37.646594Z",
                        "category": "manual",
                    },
                    {
                        "eval_result": {"label": "valid", "category": "Travel booking"},
                        "eval_timestamp": "2026-08-05T04:41:38.636182Z",
                        "category": "llm",
                    },
                ],
                "summary": {"eval_count": 2, "enum_counts": {"valid": 2}},
            }
        ],
        "next_page_token": None,
        "prev_page_token": None,
    }

    def test_live_response_yields_one_case(self, post):
        _queue(post, self.LIVE_RESPONSE)

        cases = OkahuEval.get_test_cases(**WINDOW, category=["manual", "llm"],
                                         eval_name="user_input_validity")

        assert cases == [FluentTestCase(
            name="9ade6084ba144b138090d64d1a082450",
            input=FactID(fact_id="9ade6084ba144b138090d64d1a082450",
                         fact_name="traces", source="okahu"),
            evals=[Eval(name="user_input_validity", result="valid")])]

    def test_null_next_page_token_stops_paging(self, post):
        _queue(post, self.LIVE_RESPONSE)

        OkahuEval.get_test_cases(**WINDOW)

        assert len(post["calls"]) == 1

    def test_latest_fallback_also_reads_eval_result(self, post):
        row = {**self.LIVE_RESPONSE["results"][0]}
        del row["authoritative"]
        _queue(post, {**self.LIVE_RESPONSE, "results": [row]})

        cases = OkahuEval.get_test_cases(**WINDOW)

        assert cases[0].evals == [Eval(name="user_input_validity", result="valid")]


class TestEmptyEndpointFallsBackToProd:
    """An empty OKAHU_API_ENDPOINT must fall back to prod, not build a hostless URL.

    tests/integration/__init__.py does os.environ.setdefault("OKAHU_API_ENDPOINT", ""),
    so the variable is *set but empty* for every integration run. os.getenv(name,
    DEFAULT) returns "" in that case -- the default only applies when the name is
    absent -- which produced the hostless URL "/api/v1/workflows/.../evals/report"
    and a MissingSchema error.
    """

    def test_empty_api_endpoint_uses_prod_host(self, post, monkeypatch):
        monkeypatch.setenv("OKAHU_API_ENDPOINT", "")
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW)

        assert post["calls"][0]["url"] == (
            "https://api.okahu.co/api/v1/workflows/wf/evals/report")

    def test_unset_api_endpoint_uses_prod_host(self, post, monkeypatch):
        monkeypatch.delenv("OKAHU_API_ENDPOINT", raising=False)
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW)

        assert post["calls"][0]["url"].startswith("https://api.okahu.co/api/")

    def test_explicit_api_endpoint_still_wins(self, post, monkeypatch):
        monkeypatch.setenv("OKAHU_API_ENDPOINT", "https://api-stage.okahu.co")
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW)

        assert post["calls"][0]["url"].startswith("https://api-stage.okahu.co/api/")

    def test_empty_eval_endpoint_uses_prod_too(self, monkeypatch):
        from monocle_test_tools.evals.okahu_filtered_eval import OkahuFilteredEval

        monkeypatch.setenv("OKAHU_EVALUATION_ENDPOINT", "")

        assert OkahuFilteredEval.from_env().eval_base == "https://eval.okahu.co/api"


class TestSpanLoaderEmptyEndpoint:
    """OkahuSpanLoader has the same set-but-empty trap as OkahuFilteredEval.from_env.

    It bites the step right after discovery: with_trace_source(testcase=...) loads
    the fact's spans through this base url.
    """

    def test_empty_api_endpoint_uses_prod_host(self, monkeypatch):
        from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

        monkeypatch.setenv("OKAHU_API_ENDPOINT", "")

        assert OkahuSpanLoader._get_api_base() == "https://api.okahu.co"

    def test_explicit_endpoint_argument_still_wins(self, monkeypatch):
        from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

        monkeypatch.setenv("OKAHU_API_ENDPOINT", "")

        assert OkahuSpanLoader._get_api_base("https://api-stage.okahu.co/") == (
            "https://api-stage.okahu.co")

    def test_env_endpoint_still_wins_when_set(self, monkeypatch):
        from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

        monkeypatch.setenv("OKAHU_API_ENDPOINT", "https://api-stage.okahu.co")

        assert OkahuSpanLoader._get_api_base() == "https://api-stage.okahu.co"
