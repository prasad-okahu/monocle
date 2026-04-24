from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES, SPAN_SUBTYPES
from monocle_apptrace.instrumentation.common.utils import get_span_id, get_error_message
from monocle_apptrace.instrumentation.metamodel.claude_cli import _helper

REQUEST = {
    "type": SPAN_TYPES.AGENTIC_REQUEST,
    "subtype": SPAN_SUBTYPES.TURN,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: _helper.get_agent_type(arguments)},
            {"attribute": "name", "accessor": lambda arguments: _helper.get_agent_name(arguments)},
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.extract_agent_request_input(arguments),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_response(arguments["result"]),
                },
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments),
                },
            ],
        },
    ],
}

INVOCATION = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: _helper.get_agent_type(arguments)},
            {"attribute": "name", "accessor": lambda arguments: _helper.get_agent_name(arguments)},
            {"attribute": "from_agent_span_id", "accessor": lambda arguments: get_span_id(arguments["parent_span"])},
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: arguments["kwargs"].get("prompt", ""),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_response(arguments["result"]),
                },
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments),
                },
            ],
        },
    ],
}

SUBAGENT_INVOCATION = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: _helper.get_agent_type(arguments)},
            {
                "attribute": "name",
                "accessor": lambda arguments: arguments["kwargs"].get("agent_type", "agent"),
            },
            {
                "attribute": "description",
                "accessor": lambda arguments: arguments["kwargs"].get("description", ""),
            },
            {"attribute": "from_agent", "accessor": lambda arguments: _helper.get_agent_name(arguments)},
            {"attribute": "from_agent_span_id", "accessor": lambda arguments: get_span_id(arguments["parent_span"])},
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: arguments["kwargs"].get("prompt", ""),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_agent_response(arguments["result"]),
                }
            ],
        },
    ],
}
