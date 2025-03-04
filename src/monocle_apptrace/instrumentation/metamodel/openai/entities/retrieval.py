from monocle_apptrace.instrumentation.metamodel.openai import (
    _helper,
)
from monocle_apptrace.instrumentation.common.utils import resolve_from_alias

RETRIEVAL = {
    "type": "retrieval",
    "attributes": [
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'model.embedding.' + resolve_from_alias(arguments['kwargs'], ['model', 'model_name', 'endpoint_name', 'deployment_name'])
            }
        ]
    ],
    "events": [

    ]
}
