"""Live end-to-end eval regression driven entirely by get_test_cases() (opt-in).

The whole test suite is derived from what Okahu already recorded: get_test_cases
discovers every fact in the window that carries a `hallucination` eval and turns
each into a FluentTestCase whose input is the fact and whose expected result is
the label already on it. Re-running those evals is then a regression check --
it catches an eval template change that would re-label previously-graded facts.

Nothing here is hand-written per fact, so the suite grows and shrinks with the
recorded data rather than with edits to this file.

Run against prod:

    OKAHU_API_KEY=... RUN_LIVE_GET_TEST_CASES=1 \
      pytest test_tools/tests/integration/test_get_test_cases_live.py -v

Against stage, add the endpoint overrides:

    OKAHU_API_ENDPOINT=https://api-stage.okahu.co \
    OKAHU_EVALUATION_ENDPOINT=https://evals-stage.okahu.co/api
"""
import os

import pytest

from monocle_test_tools import get_test_cases
from monocle_test_tools.pytest_plugin import monocle_trace_asserter  # noqa: F401 (fixture)

WORKFLOW = "test_cc_customer_care_agent"
START_TIME = "2026-05-26T04:27:41.907161Z"
END_TIME = "2026-05-27T04:27:41.907161Z"
EVAL_NAME = "hallucination"

# Opt-in: this makes real Okahu calls, one at collection time and one per fact.
_ENABLED = bool(os.getenv("OKAHU_API_KEY") and os.getenv("RUN_LIVE_GET_TEST_CASES"))

pytestmark = pytest.mark.skipif(
    not _ENABLED,
    reason="requires OKAHU_API_KEY and RUN_LIVE_GET_TEST_CASES=1")


def _discover():
    """The recorded facts to re-evaluate, or [] when the live gate is off.

    Runs at import time because parametrize needs the list before collection.
    The gate is checked here too -- pytestmark only skips the *tests*, it does
    not stop this module body from executing. When the gate is on, an API error
    is deliberately left to propagate: the run was explicitly asked for, so a
    collection error is the honest outcome rather than a silently empty suite.
    """
    if not _ENABLED:
        return []
    return get_test_cases(
        source="okahu",
        workflow_name=WORKFLOW,
        start_time=START_TIME,
        end_time=END_TIME,
        eval_name=EVAL_NAME,
    )


TEST_CASES = _discover()


def test_discovery_returned_cases():
    """Guard against a vacuously green file.

    An empty parametrize list is reported as a skip, so without this the whole
    suite would look fine when discovery silently found nothing -- which is
    exactly how the authoritative/eval_result label bug hid.
    """
    assert TEST_CASES, (
        f"no facts with a '{EVAL_NAME}' eval found for workflow '{WORKFLOW}' "
        f"between {START_TIME} and {END_TIME}; nothing was regression-tested")


@pytest.mark.parametrize("testcase", TEST_CASES, ids=lambda tc: tc.name)
def test_recorded_evals_still_reproduce(monocle_trace_asserter, testcase):
    """Each discovered fact still evaluates to the label already recorded on it.

    with_trace_source loads the fact's spans (check_eval asserts on spans, and
    raises when none are in scope) and must run before with_evaluation, which
    reads the trace source the former sets. The FactID on the test case supplies
    the id and fact level, so only workflow_name is passed explicitly.
    """
    (monocle_trace_asserter
        .with_trace_source(testcase=testcase, workflow_name=WORKFLOW)
        .with_evaluation("okahu")
        .check_eval(testcase=testcase))
