

import json
import logging
import os.path
import time
import unittest
from typing import List
from unittest.mock import ANY, patch

import requests
from common.helpers import OurLLM
from common.http_span_exporter import HttpSpanExporter
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod

logger = logging.getLogger(__name__)
LLAMA_INDEX_RETRIEVAL_SPAN_NAME = "llama_index.core.indices.vector_store.retrievers.retriever.VectorIndexRetriever"
LLAMA_INDEX_QUERY_SPAN_NAME = "llama_index.core.query_engine.retriever_query_engine.RetrieverQueryEngine"
class TestHandler(unittest.TestCase):

    instrumentor = None

    def setUp(self):
        os.environ["HTTP_API_KEY"] = "key1"
        os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"
        self.instrumentor = setup_monocle_telemetry(
            workflow_name="llama_index_1",
            span_processors=[
                    BatchSpanProcessor(HttpSpanExporter(os.environ["HTTP_INGESTION_ENDPOINT"])),
                    BatchSpanProcessor(ConsoleSpanExporter())
                ],
            wrapper_methods=[
                        WrapperMethod(
                            package="helpers",
                            object_name="OurLLM",
                            method="complete",
                            span_name="llamaindex.OurLLM",
                            output_processor="output_processor"
                            ),
                    ])
        

    def tearDown(self) -> None:
        try:
            if self.instrumentor is not None:
                self.instrumentor.uninstrument()
        except Exception as e:
            print("Uninstrument failed:", e)
        return super().tearDown()

    @patch.object(requests.Session, 'post')
    def test_llama_index(self, mock_post):
        
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = 'mock response'

       

        llm = OurLLM()

        # check if storage already exists
        PERSIST_DIR = "../storage"
        if not os.path.exists(PERSIST_DIR):
            # load the documents and create the index
            dir_path = os.path.dirname(os.path.realpath(__file__))
            documents = SimpleDirectoryReader(os.path.join(dir_path, "..", "data")).load_data()
            index = VectorStoreIndex.from_documents(documents)
            # store it for later
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)


        # Either way we can now query the index
        query_engine = index.as_query_engine(llm= llm)
        query = "What did the author do growing up?"
        response = query_engine.query(query)
        time.sleep(6)
        mock_post.assert_called_with(
            url = os.environ["HTTP_INGESTION_ENDPOINT"],
            data=ANY,
            timeout=ANY
        )
        '''mock_post.call_args gives the parameters used to make post call.
           This can be used to do more asserts'''
        dataBodyStr = mock_post.call_args.kwargs['data']
        logger.debug(dataBodyStr)
        dataJson = json.loads(dataBodyStr) # more asserts can be added on individual fields

        root_span = [x for x in  dataJson["batch"] if(x["parent_id"] == "None")][0]
        root_span_events = root_span["events"]

        def get_event_attributes(events, key):
            return [event['attributes'] for event in events if event['name'] == key][0]

        # input_event_attributes = get_event_attributes(root_span_events, PROMPT_INPUT_KEY)
        # output_event_attributes = get_event_attributes(root_span_events, PROMPT_OUTPUT_KEY)
        #
        # assert input_event_attributes[QUERY] == query
        # assert output_event_attributes[RESPONSE] == llm.dummy_response

        span_names: List[str] = [span["name"] for span in dataJson['batch']]
        llm_span = [x for x in  dataJson["batch"] if "llamaindex.OurLLM" in x["name"]]
        if llm_span:
            assert llm_span[0]['events'][2]["attributes"]["completion_tokens"] == 1
            assert llm_span[0]['events'][2]["attributes"]["prompt_tokens"] == 2
            assert llm_span[0]['events'][2]["attributes"]["total_tokens"] == 3
        vectorstore_retriever_span = [x for x in  dataJson["batch"] if LLAMA_INDEX_RETRIEVAL_SPAN_NAME in x["name"]][0]
        for name in [LLAMA_INDEX_RETRIEVAL_SPAN_NAME, LLAMA_INDEX_QUERY_SPAN_NAME]:
            assert name in span_names
        assert vectorstore_retriever_span["attributes"]['entity.1.name'] == "SimpleVectorStore"
        assert vectorstore_retriever_span["attributes"]['entity.1.type'] == 'vectorstore.SimpleVectorStore'
        assert vectorstore_retriever_span["attributes"]['entity.2.name'] == "BAAI/bge-small-en-v1.5"
        assert vectorstore_retriever_span["attributes"]['entity.2.type'] == 'model.embedding.BAAI/bge-small-en-v1.5'


        type_found = False
        vectorstore_provider = False

        for span in dataJson["batch"]:
            if span["name"] == "workflow":
                assert span["attributes"]["entity.1.type"] == "workflow.llamaindex"
                type_found = True
            if span["name"] == LLAMA_INDEX_RETRIEVAL_SPAN_NAME:
                assert span["attributes"]['entity.1.name'] == "SimpleVectorStore"
                vectorstore_provider = True
        assert type_found
        assert vectorstore_provider



if __name__ == '__main__':
    unittest.main()

