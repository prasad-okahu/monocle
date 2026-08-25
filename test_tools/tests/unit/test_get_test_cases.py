"""Building FluentTestCases from the traces recorded for an Okahu workflow.

get_trace_ids enumerates the window -- those traces ARE the test cases. Only when
an eval_name is given is /evals/report asked about exactly those fact_ids, and
its per-(fact_id, eval_name) rows become each case's expected results. Passing
fact_ids takes that endpoint out of discovery mode, whose selector was their
absence. Either way the cases feed straight into the testcase= plumbing on
with_trace_source / run_agent / check_eval.
"""
import os

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
EVAL = "hallucination"

# The traces get_trace_ids reports for the window. Test cases now come from this
# list and the eval report only enriches it, so every report-focused test needs
# some traces to exist. Tests that care about the set override TRACE_IDS[:].
TRACE_IDS = ["aaa", "bbb", "9ade6084ba144b138090d64d1a082450"]


@pytest.fixture(autouse=True)
def _stub_get_spans(monkeypatch):
    """Every trace is now fetched, so keep that off the network by default.

    Returns no spans, which from_spans turns into a case with empty agents and
    tools -- all the report-focused tests need. TestPopulatedFromSpans overrides
    this with a real recorded trace.
    """
    from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

    monkeypatch.setattr(OkahuSpanLoader, "get_spans", staticmethod(lambda *a, **k: []))


@pytest.fixture(name="stub_traces")
def stub_traces_fixture(monkeypatch):
    """Stand in for the trace enumeration that now precedes every report call.

    Explicit rather than autouse: TestTraceIdEnumeration and
    TestOptionalFactFilter exercise get_trace_ids itself and must not have it
    stubbed out from under them.
    """
    from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

    monkeypatch.setattr(OkahuSpanLoader, "get_trace_ids",
                        staticmethod(lambda *a, **k: list(TRACE_IDS)))


@pytest.mark.usefixtures("stub_traces")
class TestRequestBody:

    def test_posts_to_the_report_path(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert post["calls"][0]["url"] == (
            "https://api.example/api/v1/workflows/wf/evals/report")

    def test_sends_the_api_key(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert post["calls"][0]["headers"]["x-api-key"] == "test-key"

    def test_body_carries_the_required_window(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        body = post["calls"][0]["body"]
        assert body["fact_name"] == "traces"
        assert body["start_time"] == "2026-05-01"
        assert body["end_time"] == "2026-06-30"

    def test_sends_the_enumerated_fact_ids(self, post):
        """fact_ids takes the endpoint OUT of discovery mode: it reports on the
        traces get_trace_ids already found rather than re-discovering them."""
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert post["calls"][0]["body"]["fact_ids"] == TRACE_IDS

    def test_category_defaults_to_llm_and_manual(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert post["calls"][0]["body"]["category"] == ["llm", "manual"]

    def test_category_string_is_wrapped(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL, category="manual")

        assert post["calls"][0]["body"]["category"] == ["manual"]

    def test_category_list_passes_through(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL, category=["llm", "manual"])

        assert post["calls"][0]["body"]["category"] == ["llm", "manual"]

    def test_fact_name_is_mapped_to_the_okahu_name(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL, fact_name="agentic_turns")

        assert post["calls"][0]["body"]["fact_name"] == "agent_requests"

    def test_no_report_call_at_all_without_an_eval_name(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW)

        assert post["calls"] == []

    def test_eval_name_becomes_a_one_item_eval_names(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name="hallucination")

        assert post["calls"][0]["body"]["eval_names"] == ["hallucination"]

    def test_custom_eval_name_is_rejected(self, post):
        with pytest.raises(ValueError, match="custom"):
            OkahuEval.get_test_cases(**WINDOW, eval_name="custom")

    def test_page_size_is_sent(self, post):
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL, page_size=250)

        assert post["calls"][0]["body"]["page_size"] == 250


@pytest.mark.usefixtures("stub_traces")
class TestPagination:

    def test_follows_the_next_page_token(self, post):
        _queue(post,
               {"results": [_row("aaa", "hallucination", "minor")],
                "next_page_token": "tok-2"},
               {"results": [_row("bbb", "hallucination", "major")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert len(post["calls"]) == 2
        assert post["calls"][1]["body"]["page_token"] == "tok-2"
        assert [c.name for c in cases] == ["aaa", "bbb"]

    def test_stops_when_no_next_page_token(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor")]})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert len(post["calls"]) == 1

    def test_empty_next_page_token_stops(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor")],
                      "next_page_token": ""})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert len(post["calls"]) == 1


@pytest.mark.usefixtures("stub_traces")
class TestMapping:

    def test_one_case_per_fact_with_a_factid_input(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor_hallucination")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert cases == [FluentTestCase(
            name="aaa",
            input=FactID(fact_id="aaa", fact_name="traces", source="okahu"),
            evals=[Eval(name="hallucination", result="minor_hallucination")])]

    def test_several_evals_group_onto_one_fact(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor"),
                                  _row("aaa", "sentiment", "positive")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert len(cases) == 1
        assert cases[0].evals == [Eval(name="hallucination", result="minor"),
                                  Eval(name="sentiment", result="positive")]

    def test_distinct_facts_become_distinct_cases(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor"),
                                  _row("bbb", "hallucination", "major")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert [c.name for c in cases] == ["aaa", "bbb"]

    def test_fact_id_is_normalized_to_bare_hex(self, post):
        _queue(post, {"results": [_row("0xaaa", "hallucination", "minor")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert cases[0].input.fact_id == "aaa"

    def test_user_facing_fact_name_is_kept_on_the_factid(self, post):
        """The body is sent the mapped name; the FactID keeps the caller's."""
        _queue(post, {"results": [_row("aaa", "hallucination", "minor")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL, fact_name="agentic_turns")

        assert cases[0].input.fact_name == "agentic_turns"

    def test_falls_back_to_latest_when_no_authoritative(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", latest_label="major")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert cases[0].evals == [Eval(name="hallucination", result="major")]

    def test_authoritative_wins_over_latest(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor",
                                       latest_label="major")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert cases[0].evals == [Eval(name="hallucination", result="minor")]

    def test_unlabeled_row_is_skipped(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor"),
                                  _row("aaa", "sentiment")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert cases[0].evals == [Eval(name="hallucination", result="minor")]

    def test_fact_with_no_labels_produces_no_case(self, post):
        """An empty evals list would raise in check_eval -- emit nothing instead."""
        _queue(post, {"results": [_row("aaa", "hallucination")]})

        assert OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL) == []

    def test_eval_not_found_row_is_skipped(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor",
                                       eval_found=False)]})

        assert OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL) == []

    def test_no_results_gives_no_cases(self, post):
        _queue(post, {"results": []})

        assert OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL) == []


@pytest.mark.usefixtures("stub_traces")
class TestFluentEntryPoint:

    def test_delegates_to_okahu(self, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor")]})

        cases = get_test_cases(source="okahu", eval_name=EVAL, **WINDOW)

        assert [c.name for c in cases] == ["aaa"]

    def test_eval_name_is_forwarded(self, post):
        _queue(post, {"results": []})

        get_test_cases(source="okahu", eval_name="hallucination", **WINDOW)

        assert post["calls"][0]["body"]["eval_names"] == ["hallucination"]

    def test_defaults_to_okahu(self, post):
        _queue(post, {"results": []})

        get_test_cases(eval_name=EVAL, **WINDOW)

        assert post["calls"][0]["url"].endswith("/evals/report")

    @pytest.mark.parametrize("source", ["file", "s3", "memory"])
    def test_unsupported_sources_are_rejected(self, source):
        """'local' is supported (see TestLocalSource); these are not."""
        with pytest.raises(ValueError, match="does not support source"):
            get_test_cases(source=source, **WINDOW)


def test_exported_from_the_package():
    import monocle_test_tools

    assert monocle_test_tools.get_test_cases is get_test_cases


@pytest.mark.usefixtures("stub_traces")
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

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert len(post["calls"]) == 1

    def test_latest_fallback_also_reads_eval_result(self, post):
        row = {**self.LIVE_RESPONSE["results"][0]}
        del row["authoritative"]
        _queue(post, {**self.LIVE_RESPONSE, "results": [row]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert cases[0].evals == [Eval(name="user_input_validity", result="valid")]


@pytest.mark.usefixtures("stub_traces")
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

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert post["calls"][0]["url"] == (
            "https://api.okahu.co/api/v1/workflows/wf/evals/report")

    def test_unset_api_endpoint_uses_prod_host(self, post, monkeypatch):
        monkeypatch.delenv("OKAHU_API_ENDPOINT", raising=False)
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert post["calls"][0]["url"].startswith("https://api.okahu.co/api/")

    def test_explicit_api_endpoint_still_wins(self, post, monkeypatch):
        monkeypatch.setenv("OKAHU_API_ENDPOINT", "https://api-stage.okahu.co")
        _queue(post, {"results": []})

        OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

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


@pytest.mark.usefixtures("stub_traces")
class TestLocalSource:
    """get_test_cases(source="local", path=...) loads a committed JSON array.

    The point is the golden-dataset workflow: freeze what Okahu returned today,
    commit it, and re-run it later with no network call.
    """

    def _write(self, tmp_path, payload):
        import json
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_loads_an_array_of_cases(self, tmp_path):
        path = self._write(tmp_path, [
            {"input": {"fact_id": "aaa", "fact_name": "traces", "source": "okahu"},
             "expected": {"evals": {"hallucination": "major_hallucination"}}},
            {"input": {"fact_id": "bbb", "fact_name": "traces", "source": "okahu"},
             "expected": {"evals": {"hallucination": "no_hallucination"}}},
        ])

        cases = get_test_cases(source="local", path=path)

        assert [c.input.fact_id for c in cases] == ["aaa", "bbb"]
        assert cases[0].evals == [Eval(name="hallucination",
                                       result="major_hallucination")]

    def test_accepts_the_same_shapes_a_parametrize_literal_does(self, tmp_path):
        """expected wrapper, evals as a mapping, scalar input -- all normalized."""
        path = self._write(tmp_path, [
            {"name": "t1", "input": "Book a flight",
             "expected": {"evals": {"hallucination": "minor", "sentiment": "positive"},
                          "token_limit": 5000}},
        ])

        cases = get_test_cases(source="local", path=path)

        assert cases[0].input == ("Book a flight",)
        assert cases[0].token_limit == 5000
        assert [e.name for e in cases[0].evals] == ["hallucination", "sentiment"]

    def test_round_trips_a_dumped_discovery_result(self, tmp_path, post):
        """Discover -> dump -> load must produce an equal set of cases."""
        _queue(post, {"results": [_row("aaa", "hallucination", "minor_hallucination"),
                                  _row("bbb", "sentiment", "positive")]})
        discovered = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)
        path = self._write(tmp_path, [c.model_dump() for c in discovered])

        assert get_test_cases(source="local", path=path) == discovered

    def test_makes_no_network_call(self, tmp_path, post):
        path = self._write(tmp_path, [{"evals": {"hallucination": "minor"}}])

        get_test_cases(source="local", path=path)

        assert post["calls"] == []

    def test_empty_array_gives_no_cases(self, tmp_path):
        assert get_test_cases(source="local", path=self._write(tmp_path, [])) == []

    def test_path_is_required(self):
        with pytest.raises(ValueError, match="'path' is required"):
            get_test_cases(source="local")

    def test_missing_file_raises_naming_the_path(self, tmp_path):
        missing = str(tmp_path / "nope.json")

        with pytest.raises(FileNotFoundError, match="nope.json"):
            get_test_cases(source="local", path=missing)

    def test_invalid_json_raises_naming_the_path(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(ValueError, match="bad.json"):
            get_test_cases(source="local", path=str(path))

    def test_top_level_object_is_rejected(self, tmp_path):
        path = self._write(tmp_path, {"evals": {"hallucination": "minor"}})

        with pytest.raises(ValueError, match="array of test cases"):
            get_test_cases(source="local", path=path)

    def test_a_bad_element_names_its_index(self, tmp_path):
        path = self._write(tmp_path, [
            {"evals": {"hallucination": "minor"}},
            {"evels": {"hallucination": "minor"}},
        ])

        with pytest.raises(ValueError, match=r"test case 1\b"):
            get_test_cases(source="local", path=path)



class TestTraceIdEnumeration:
    """Test cases come from get_trace_ids; the eval report only enriches them.

    Passing fact_ids takes /evals/report out of discovery mode -- the absence of
    fact_ids is what selected discovery -- so this is a targeted report over the
    traces the window actually contains.
    """

    @pytest.fixture(name="traces")
    def traces_fixture(self, monkeypatch):
        """Stub get_trace_ids, recording its kwargs."""
        from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

        seen = {"ids": ["aaa", "bbb"]}

        def fake(workflow_name, fact_name=None, fact_id=None, **kwargs):
            seen["call"] = {"workflow_name": workflow_name, "fact_name": fact_name,
                            "fact_id": fact_id, **kwargs}
            return seen["ids"]

        monkeypatch.setattr(OkahuSpanLoader, "get_trace_ids", staticmethod(fake))
        return seen

    def test_one_case_per_trace_id_without_eval_name(self, traces, post):
        cases = OkahuEval.get_test_cases(**WINDOW)

        assert [c.name for c in cases] == ["aaa", "bbb"]
        assert [c.input.fact_id for c in cases] == ["aaa", "bbb"]

    def test_no_report_call_without_eval_name(self, traces, post):
        OkahuEval.get_test_cases(**WINDOW)

        assert post["calls"] == []

    def test_cases_without_eval_name_carry_no_evals(self, traces, post):
        cases = OkahuEval.get_test_cases(**WINDOW)

        assert all(c.evals == [] for c in cases)

    def test_trace_lookup_gets_the_window(self, traces, post):
        OkahuEval.get_test_cases(**WINDOW)

        assert traces["call"]["start_time"] == "2026-05-01"
        assert traces["call"]["end_time"] == "2026-06-30"

    def test_trace_lookup_sends_no_fact_filter(self, traces, post):
        OkahuEval.get_test_cases(**WINDOW)

        assert traces["call"]["fact_name"] is None
        assert traces["call"]["fact_id"] is None

    def test_report_receives_the_fact_ids(self, traces, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor")]})

        OkahuEval.get_test_cases(**WINDOW, eval_name="hallucination")

        assert post["calls"][0]["body"]["fact_ids"] == ["aaa", "bbb"]
        assert post["calls"][0]["body"]["eval_names"] == ["hallucination"]

    def test_evals_are_attached_to_their_fact(self, traces, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor"),
                                  _row("bbb", "hallucination", "major")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name="hallucination")

        assert [(c.name, c.evals[0].result) for c in cases] == [
            ("aaa", "minor"), ("bbb", "major")]

    def test_a_trace_with_no_labelled_eval_is_dropped(self, traces, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name="hallucination")

        assert [c.name for c in cases] == ["aaa"]

    def test_trace_order_is_preserved(self, traces, post):
        traces["ids"] = ["ccc", "aaa", "bbb"]
        _queue(post, {"results": [_row("aaa", "hallucination", "minor"),
                                  _row("ccc", "hallucination", "minor"),
                                  _row("bbb", "hallucination", "minor")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name="hallucination")

        assert [c.name for c in cases] == ["ccc", "aaa", "bbb"]

    def test_no_traces_gives_no_cases_and_no_report_call(self, traces, post):
        traces["ids"] = []

        assert OkahuEval.get_test_cases(**WINDOW, eval_name="hallucination") == []
        assert post["calls"] == []


class TestOptionalFactFilter:
    """get_trace_ids can now enumerate a window with no fact filter."""

    @pytest.fixture(name="get")
    def get_fixture(self, monkeypatch):
        from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

        seen = {}

        def fake_do_get(url, headers, params=None, timeout=30, context_msg=""):
            seen["params"] = params
            return []

        monkeypatch.setattr(OkahuSpanLoader, "_do_get", staticmethod(fake_do_get))
        return seen

    def test_no_fact_filter_omits_both_params(self, get):
        from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

        OkahuSpanLoader.get_trace_ids("wf", start_time="a", end_time="b")

        assert "duration_fact" not in get["params"]
        assert "fact_ids" not in get["params"]
        assert get["params"]["start_time"] == "a"

    def test_a_fact_filter_is_still_sent(self, get):
        from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

        OkahuSpanLoader.get_trace_ids("wf", "agent_sessions", "sess_1")

        assert get["params"]["duration_fact"] == "agent_sessions"
        assert get["params"]["fact_ids"] == "sess_1"

    @pytest.mark.parametrize("half", [
        {"fact_name": "agent_sessions"}, {"fact_id": "sess_1"},
    ])
    def test_half_a_filter_raises(self, get, half):
        from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

        with pytest.raises(ValueError, match="fact_name and fact_id"):
            OkahuSpanLoader.get_trace_ids("wf", **half)


class TestPopulatedFromSpans:
    """Each trace's spans fill in the case: agents, tools, token_limit.

    from_spans does the reading; get_test_cases supplies name, the FactID input
    and the evals, so the result is a fully described case rather than a bare
    pointer at a fact.
    """

    TRACE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "traces", "trace1.json")
    # from_spans orders by span start_time, i.e. call order -- the supervisor
    # is invoked first and delegates to the rest.
    AGENTS = ["adk_supervisor_agent_5", "adk_flight_booking_agent_5",
              "adk_hotel_booking_agent_5", "adk_trip_summary_agent_5"]

    @pytest.fixture(name="spans")
    def spans_fixture(self, monkeypatch):
        """Stub get_spans with a real recorded trace, recording its calls."""
        from monocle_test_tools.okahu_span_loader import OkahuSpanLoader
        from monocle_test_tools.span_loader import JSONSpanLoader

        loaded = JSONSpanLoader.from_json(self.TRACE)
        calls = []

        def fake(workflow_name, trace_id, **kwargs):
            calls.append({"workflow_name": workflow_name, "trace_id": trace_id, **kwargs})
            return loaded

        monkeypatch.setattr(OkahuSpanLoader, "get_spans", staticmethod(fake))
        monkeypatch.setattr(OkahuSpanLoader, "get_trace_ids",
                            staticmethod(lambda *a, **k: ["aaa", "bbb"]))
        return calls

    def test_get_spans_is_called_once_per_trace(self, spans, post):
        OkahuEval.get_test_cases(**WINDOW)

        assert [c["trace_id"] for c in spans] == ["aaa", "bbb"]

    def test_get_spans_gets_the_window(self, spans, post):
        OkahuEval.get_test_cases(**WINDOW)

        assert spans[0]["start_time"] == "2026-05-01"
        assert spans[0]["end_time"] == "2026-06-30"

    def test_agents_are_populated(self, spans, post):
        cases = OkahuEval.get_test_cases(**WINDOW)

        assert [a.name for a in cases[0].agents] == self.AGENTS

    def test_agents_carry_their_recorded_output(self, spans, post):
        cases = OkahuEval.get_test_cases(**WINDOW)

        hotel = [a for a in cases[0].agents
                 if a.name == "adk_hotel_booking_agent_5"][0]
        assert hotel.output.startswith("OK. I have booked a stay")

    def test_tools_are_populated(self, spans, post):
        cases = OkahuEval.get_test_cases(**WINDOW)

        assert [t.name for t in cases[0].tools] == ["adk_book_hotel_5"]

    def test_token_limit_is_populated(self, spans, post):
        cases = OkahuEval.get_test_cases(**WINDOW)

        assert cases[0].token_limit and cases[0].token_limit > 0

    def test_input_is_the_factid_not_the_recorded_prompt(self, spans, post):
        """with_trace_source(testcase=) needs a FactID; run_agent resolves it."""
        cases = OkahuEval.get_test_cases(**WINDOW)

        assert cases[0].input == FactID(fact_id="aaa", fact_name="traces",
                                        source="okahu")

    def test_name_is_the_fact_id_not_the_workflow(self, spans, post):
        cases = OkahuEval.get_test_cases(**WINDOW)

        assert [c.name for c in cases] == ["aaa", "bbb"]

    def test_evals_are_attached_alongside_the_agents(self, spans, post):
        _queue(post, {"results": [_row("aaa", "hallucination", "minor"),
                                  _row("bbb", "hallucination", "major")]})

        cases = OkahuEval.get_test_cases(**WINDOW, eval_name=EVAL)

        assert cases[0].evals == [Eval(name="hallucination", result="minor")]
        assert [a.name for a in cases[0].agents] == self.AGENTS

    def test_no_traces_means_no_get_spans_calls(self, monkeypatch, spans, post):
        from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

        monkeypatch.setattr(OkahuSpanLoader, "get_trace_ids",
                            staticmethod(lambda *a, **k: []))

        assert OkahuEval.get_test_cases(**WINDOW) == []
        assert spans == []
