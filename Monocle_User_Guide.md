#Monocle User Guide

## Monocle Concepts
### Traces
Traces are the full view of a single end-to-end application KPI, for example Chatbot application to provide a response to end user’s question. Traces consist of various metadata about the application run including status, start time, duration, input/outputs etc. They also include a list of individual steps aka “spans with details about that step.
It’s typically the workflow code components of an application that generate the traces for application runs. 
### Spans
Spans are the individual steps executed by the application to perform a GenAI related task”, for example app retrieving vectors from DB, app querying LLM for inference etc. The span includes the type of operation, start time, duration and metadata relevant to that step e.g., Model name, parameters and model endpoint/server for an inference request.
It’s typically the workflow code components of an application that generate the traces for application runs.

## Setup Monocle
- You can download Monocle library releases from Pypi
``` 
    > pip install monocle_apptrace
```
- You can locally build and install Monocle library from source
```
> pip install .
```
- Install the optional test dependencies listed against dev in pyproject.toml in editable mode
```
> pip install -e ".[dev]"
```

## Examples 
### Enable Monocle tracing in your application
```python
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Call the setup Monocle telemetry method
setup_monocle_telemetry(workflow_name = "simple_math_app",
        span_processors=[BatchSpanProcessor(ConsoleSpanExporter())])

llm = OpenAI()
prompt = PromptTemplate.from_template("1 + {number} = ")

chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"number":2})

# Request callbacks: Finally, let's use the request `callbacks` to achieve the same result
chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"number":2}, {"callbacks":[handler]})
    
```

### Monitoring custom methods with Monocle

```python
from monocle_apptrace.wrapper import WrapperMethod,task_wrapper,atask_wrapper
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# extend the default wrapped methods list as follows
app_name = "simple_math_app"
setup_monocle_telemetry(
        workflow_name=app_name,
        span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
        wrapper_methods=[
            WrapperMethod(
                package="langchain.schema.runnable",
                object="RunnableParallel",
                method="invoke",
                span_name="langchain.workflow",
                wrapper=task_wrapper),
            WrapperMethod(
                package="langchain.schema.runnable",
                object="RunnableParallel",
                method="ainvoke",
                span_name="langchain.workflow",
                wrapper=atask_wrapper)
        ])

```

### Default configuration of instrumented methods in Monocle

The following files comprise of default configuration of instrumented methods and span names corresponding to them, for each framework respectively. 

[src/monocle_apptrace/langchain/__init__.py](src/monocle_apptrace/langchain/__init__.py),
 [src/monocle_apptrace/llamaindex/__init__.py](src/monocle_apptrace/llamaindex/__init__.py),
 [src/monocle_apptrace/haystack/__init__.py](src/monocle_apptrace/haystack/__init__.py)

Following configuration instruments  ```invoke(..)``` of ```RunnableSequence```, aka chain or worflow in Langchain parlance, to emit the span.

```
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableSequence",
        "method": "invoke",
        "span_name": "langchain.workflow",
        "wrapper": task_wrapper
    }
```

#### span json 

Monocle generates spans which adhere to [Tracing API | OpenTelemetry](https://opentelemetry.io/docs/specs/otel/trace/api/#span) format. Please note that ```trace_id``` groups related spans and is auto generated with-in Monocle. 

| Span JSON      | Description     |
| ------------- | ------------- |
| {||
|  "```name```": "langchain.workflow",|span name and is configurable in [__init.py__](src/monocle_apptrace/langchain/__init__.py) or in ```setup_okahu_telemetry(...)```|
|  "```context```": {|this gets autogenerated|
| &ensp;    "```trace_id```": "0xe5269f0e534efa098b240f974220d6b7",||
| &ensp;       "```span_id```": "0x30b13075eca52f44",||
| &ensp;       "```trace_state```": "[]"||
| &ensp;   },||
|"```kind```": "SpanKind.INTERNAL",| an enum that describes what this span is about. Default value is SpanKind.INTERNAL, as current enums do not cover ML apps |
|"```parent_id```": null,|if null, this is root span|
|"```start_time```": "2024-07-16T17:05:15.544861Z",||
|"```end_time```": "2024-07-16T17:05:43.502007Z",||
|"```status```": {||
|&ensp;  "```status_code```": "UNSET"| status of span to OK or ERROR. Default is UNSET|
|&ensp; },||
|"```attributes```": {||
|&ensp; "workflow_name": "ml_rag_app",|defines the name of the service being set in ```setup_okahu_telemetry(...)``` during initialization of instrumentation|
|&ensp; "workflow_type": "workflow.langchain"|type of framework that generated this span|
|&ensp; },||
|"```events```": [|captures the log records|
|&ensp; {||
|&ensp;&emsp;  "```name```": "input",|name of the event. If the span is about LLM, then this will be 'input'. For vector store retrieval, this would be 'context_input'|
|&ensp;&emsp;  "```timestamp```": "2024-07-16T17:05:15.544874Z",||
|&ensp;&emsp;  "```attributes```": {|captures the 'input' attributes. Based on the workflow of the ML framework being used, the attributes change|
|&emsp;&emsp;&emsp;    "question": "What is Task Decomposition?",|represents LLM query|
|&emsp;&emsp;&emsp;    "q_a_pairs": "..." |represents questions and answers for a few shot LLM prompting |
|&emsp;&emsp;              }||
|&emsp;         },||
|&emsp; {||
|&emsp;&emsp;  "```name```": "output",|represents 'ouput' event of LLM|
|&emsp;&emsp; "```timestamp```": "2024-07-16T17:05:43.501996Z",||
|&emsp;&emsp;"```attributes```": {||
|&emsp;&emsp;&emsp; "response": "Task Decomposition is ..."|response to LLM query. |
|&emsp;&emsp;&emsp;}||
|&emsp;&emsp;}||
|&emsp;    ],||
|&emsp; "```links```": [],|unused. Ideally this links other causally-related spans,<br/> but as spans are grouped by ```trace_id```, and ```parent_id``` links to parent span, this is unused|
|&emsp;   "```resource```": {|represents the service name or server or machine or container which generated the span|
|&emsp;&emsp;&emsp;  "```attributes```": {||
|&emsp;&emsp;&emsp;&emsp;  "service.name": "ml_rag_app"|only service.name is being populated and defaults to the value of 'workflow_name' |
|&emsp;&emsp;&emsp;  },||
|&emsp;&emsp;"```schema_url```": ""|unused|
|&emsp;&emsp;     }||
|} | |