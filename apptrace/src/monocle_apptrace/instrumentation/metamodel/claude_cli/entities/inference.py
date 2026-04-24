from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES

INFERENCE = {
    "type": SPAN_TYPES.INFERENCE,
    "attributes": [
        [
            {"attribute": "type", "accessor": lambda arguments: "inference.anthropic"},
            {"attribute": "provider_name", "accessor": lambda arguments: "api.anthropic.com"},
            {"attribute": "inference_endpoint", "accessor": lambda arguments: "https://api.anthropic.com"},
        ],
        [
            {"attribute": "name", "accessor": lambda arguments: arguments["kwargs"].get("model", "claude")},
            {"attribute": "type", "accessor": lambda arguments: "model.llm." + arguments["kwargs"].get("model", "claude")},
        ],
        # Entity 3: the tool dispatched in this round (empty string = no entity created)
        [
            {"attribute": "name", "accessor": lambda arguments: arguments["kwargs"].get("tool_name", "")},
            {"attribute": "type", "accessor": lambda arguments: _tool_entity_type(arguments["kwargs"].get("tool_name", ""))},
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: arguments["kwargs"].get("input_text", ""),
                }
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: arguments["kwargs"].get("output_text", ""),
                }
            ],
        },
        {
            "name": "metadata",
            "attributes": [
                {"accessor": lambda arguments: arguments["kwargs"].get("tokens", {})},
                {
                    "attribute": "finish_reason",
                    "accessor": lambda arguments: arguments["kwargs"].get("finish_reason", ""),
                },
                {
                    "attribute": "finish_type",
                    "accessor": lambda arguments: arguments["kwargs"].get("finish_type", ""),
                },
            ],
        },
    ],
}


def _tool_entity_type(tool_name: str) -> str:
    if not tool_name:
        return ""
    if tool_name.startswith("mcp__"):
        return "tool.mcp"
    return "tool.function"
