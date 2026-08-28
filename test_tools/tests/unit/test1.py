from asyncio import sleep
import pytest

from monocle_test_tools import TraceAssertion
from test_common.adk_travel_agent import root_agent

# Case1: Baseline testing - build golden dataset
# Agent inputs are parametric, the agent behavior validation is in test code
BASELINE_TESTCASES = [
    {
        "input": "Book a flight from San Francisco to Mumbai for 26th Nov 2025",
        "expected" : {
            "output" : ["Booked flight", "Mumbai", "San Francisco"],
        }
    },
    {
        "input": "Book a flight from San Francisco to Los Angeles for 26th Nov 2025",
        "expected" : {
            "output" : ["Booked flight", "Mumbai", "Los Angeles"],
        }
    }
]
@pytest.mark.parametrize("testcase", BASELINE_TESTCASES)
def test_travel_agent_baseline(monocle_trace_asserter:TraceAssertion, testcase):
    # run agent with given input specified in the testcase
    monocle_trace_asserter.run_agent(root_agent, "google_adk", testcase=testcase)

    # assert that the output tokens in the testcase are present in the agent's output
    monocle_trace_asserter.contains_output(testcase=testcase)

    # additional assertions beyond what's in the testcase
    monocle_trace_asserter.called_tool(tool_name="flight_tool", agent_name="fligh_booking_agent")
    monocle_trace_asserter.called_agent(agent_name="supervisor")