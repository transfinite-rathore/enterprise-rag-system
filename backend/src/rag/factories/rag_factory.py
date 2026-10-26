from typing import Tuple
from ..orchestrator.rag_orchestrator import RAGOrchestrator
from ..pipelines.retriever.search.dense_search import DenseSearchRule
from ..pipelines.retriever.search.sparse_search import SparseSearchRule
from ..pipelines.retriever.search.hybrid_search import HybridSearchRule
from ..pipelines.answer_generator.answer_generator import AnswerGenerator
from ...config.qdrant_client import create_qdrant_client
from  ..factories.vectorstore.VectorStoreFactory import VSFactory
from ..factories.vectorstore.VectorStore import VectorStoreInterface
from ..pipelines.retriever.retriever import Retriever
from ..pipelines.reranker.reranker import Reranker
from  ..pipelines.reranker.reranking_strategy.cross_encoder import CrossEncoderReranker
from ..pipelines.querypipeline.processor import QueryPipeline
from ..pipelines.querypipeline.validator.EmptyQueryValidator import EmptyQueryValidator
from ..pipelines.querypipeline.validator.PrintableCharacterValidator import PrintableCharacterValidator
from ..pipelines.querypipeline.validator.validator import QueryValidator
from ..pipelines.querypipeline.corrector.corrector import QueryCorrecter
from ..pipelines.querypipeline.corrector.RemoveRepeatedPunctuationRule import RemoveRepeatedPunctuationRule
from ..pipelines.querypipeline.corrector.SpaceRule import CollapseSpacesRule
from ..pipelines.querypipeline.corrector.TrimRule import TrimWhitespaceRule
from ..pipelines.querypipeline.rewriter.llm_rewriter import LLMQueryRewriter
from ..factories.LLMProvider import ChatModelProvider
from ..model.llm_config import LLMConfig
from ..model.llm_response import LLMResponse
from ..pipelines.embedder.sparse_embedder import SparseEmbedderProvider
from ..pipelines.embedder.dense_embedder import DenseEmbedderProvider
# from ..pipelines.Embedding import EmbedderProvider
from ..pipelines.resultstore.file_result_store import RAGFileResultStore
from  ...config.config_manager import get_config

def create_rag_orchestrator() -> Tuple[VectorStoreInterface,RAGOrchestrator]:
    
    ## LLM Model Setup
    llm=get_config()["llm"]
    active_provider_name = llm["active_provider"]

    # 2. Get the settings for that specific provider
    provider_settings = llm["provider"][active_provider_name]

    # 3. Initialize your config using the nested settings
    llm_config = LLMConfig(
    provider=active_provider_name,                 # "open_router"
    model_name=provider_settings["model_name"],    # "nvidia/nemotron-3-super-120b-a12b:free"
    temperature=provider_settings["temperature"]   # 0.5
)
    config=get_config()["ingestion"]

    llm_model=ChatModelProvider(llm_config).get_model()
    
    dense_embedder=DenseEmbedderProvider()
    sparse_embedder=SparseEmbedderProvider()
    client=create_qdrant_client()
    # VSFactory(client).create_vector_store()
    vector_store = VSFactory(client).create_vector_store()

    search_strategy =  DenseSearchRule() #HybridSearchRule()

    retriever = Retriever(
        vs=vector_store,
        searching_rule=search_strategy,
        collection_name=config["collection_name"],
        top_k=get_config()["reteriver"]["reterived_chunks_count"]
    )

    reranking_strategy = CrossEncoderReranker(get_config()["paths"]["cross_encoder_path"])
    reranker = Reranker(reranking_strategy,top_k=get_config()["reteriver"]["reranked_chunks_count"])
    validator=QueryValidator([EmptyQueryValidator(),PrintableCharacterValidator()])
    corrector=QueryCorrecter([TrimWhitespaceRule(),CollapseSpacesRule(),RemoveRepeatedPunctuationRule()])
    rewriter=LLMQueryRewriter(llm_model)
    query_pipeline = QueryPipeline(validator,corrector,rewriter,dense_embedder,sparse_embedder)

    answer_generator = AnswerGenerator(llm_model,LLMResponse)

    result_store = RAGFileResultStore(get_config()["reteriver"]["file_store_dir"])
    
    return (vector_store,RAGOrchestrator(
        query_processor=query_pipeline,
        retriever=retriever,
        reranker=reranker,
        result_generator=answer_generator,
        result_saver=result_store,
        
    ))