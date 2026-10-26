import pandas as pd
import asyncio
from src.config.config_manager import get_config
from src.rag.factories.rag_factory import create_rag_orchestrator
from src.rag.model.llm_config import LLMConfig
from src.rag.factories.LLMProvider import ChatModelProvider
from src.rag.pipelines.resultstore.file_result_store import RAGFileResultStore
from src.rag.factories.judge_factory import create_LLM_judge
from src.rag.evaluation_pipeline.file_store.Relevance_verdict import RelevanceVerdictFileSave
from src.rag.evaluation_pipeline.file_store.Evaluation_result import EvaluationFileSave
from src.rag.evaluation_pipeline.evaluation_orchestrator import RAGEvaluationPipeline
from src.rag.evaluation_pipeline.reterival_evaluator import RetrievalEvaluator
from src.rag.evaluation_pipeline.reranker_evaluator import RerankerEvaluator
from src.rag.evaluation_pipeline.generator_evauator import GeneratorEvaluator
from src.rag.evaluation_pipeline.evaluation_metrics.ranking_metric.MRR import MeanReciprocalRank
from src.rag.evaluation_pipeline.evaluation_metrics.ranking_metric.hit_rate import HitRate 
from src.rag.evaluation_pipeline.evaluation_metrics.ranking_metric.NDCG import NDCGAtK
from src.rag.evaluation_pipeline.evaluation_metrics.ranking_metric.precision import PrecisionAtK
from src.rag.evaluation_pipeline.evaluation_metrics.ranking_metric.recall import RecallAtK
from src.rag.evaluation_pipeline.evaluation_metrics.generation_metric.answer_relevance.relevance_evaluator import AnswerRelevanceMetric
from src.rag.evaluation_pipeline.evaluation_metrics.generation_metric.correctness.correctness_evaluator import CorrectnessEvaluationMetric
from src.rag.evaluation_pipeline.evaluation_metrics.generation_metric.faithfullness.faithfullness_evaluator import FaithfullnessEvaluationMetric 

import traceback

import asyncio
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


config = get_config()


class RAGEvaluationRunner:

    def __init__(self):

        # Create RAG only once
        self.vector_store, self.rag = create_rag_orchestrator()

        # Create stores only once
        self.result_store = RAGFileResultStore(base_dir=get_config()["reteriver"]["file_store_dir"])
        self.verdict_store = RelevanceVerdictFileSave(dir_path=get_config()["reteriver"]["file_store_dir"])
        self.evaluation_store = EvaluationFileSave(dir_path=get_config()["reteriver"]["file_store_dir"])

        # Create judge only once
        self.judge = create_LLM_judge()

        # Create LLM only once
        self.llm_model = self._create_llm_model()

        # Create evaluation pipeline only once
        self.evaluator = self._create_evaluator()

    def _create_llm_model(self):

        llm = config["llm"]

        active_provider_name = llm["active_provider"]

        provider_settings = llm["provider"][active_provider_name]

        llm_config = LLMConfig(
            provider=active_provider_name,
            model_name=provider_settings["model_name"],
            temperature=provider_settings["temperature"],
        )

        return ChatModelProvider(llm_config).get_model()

    def _create_evaluator(self):

        retrieval_metrics = [
            PrecisionAtK(get_config()["reteriver"]["reteriver_eval_top_k"]),
            RecallAtK(get_config()["reteriver"]["reteriver_eval_top_k"]),
            MeanReciprocalRank(),
            NDCGAtK(get_config()["reteriver"]["reteriver_eval_top_k"]),
            HitRate(get_config()["reteriver"]["reteriver_eval_top_k"]),
        ]

        reranker_metrics = [
            PrecisionAtK(get_config()["reteriver"]["reranker_eval_top_k"]),
            RecallAtK(get_config()["reteriver"]["reranker_eval_top_k"]),
            MeanReciprocalRank(),
            NDCGAtK(get_config()["reteriver"]["reranker_eval_top_k"]),
            HitRate(get_config()["reteriver"]["reranker_eval_top_k"]),
        ]

        generation_metrics = [
            FaithfullnessEvaluationMetric(self.llm_model),
            CorrectnessEvaluationMetric(self.llm_model),
            AnswerRelevanceMetric(self.llm_model),
        ]

        retrieval_evaluator = RetrievalEvaluator(
            retrieval_metrics
        )

        reranker_evaluator = RerankerEvaluator(
            reranker_metrics
        )

        generator_evaluator = GeneratorEvaluator(
            generation_metrics
        )

        return RAGEvaluationPipeline(
            retrieval_evaluator,
            reranker_evaluator,
            generator_evaluator,
        )

    async def evaluate(self, query: str, human_answer: str):

        run_id = None

        try:

            # -----------------------------
            # 1. Run RAG
            # -----------------------------
            run_id, response = self.rag.invoke(
                "123",
                query,
            )

            print(f"\nRun ID: {run_id}")
            print(f"Response: {response}")

            # -----------------------------
            # 2. Load RAG result
            # -----------------------------
            payload = self.result_store.load(run_id)

            # -----------------------------
            # 3. Judge retrieved chunks
            # -----------------------------
            print("Evaluating chunks...")

            verdicts = await self.judge.judge_chunks(
                human_answer=human_answer,
                payload=payload,
            )

            self.verdict_store.save_file(
                run_id,
                verdicts,
            )

            # -----------------------------
            # 4. Run metrics
            # -----------------------------
            print("Evaluation Starting !!")

            result = await self.evaluator.run(
                payload,
                verdicts,
            )

            print(result)

            # -----------------------------
            # 5. Save evaluation
            # -----------------------------
            self.evaluation_store.save_file(
                run_id,
                result,
            )

            return result

        except Exception as e:

            print(f"Evaluation failed: {e}")
            traceback.print_exc()

            return None

    def close(self):

        if self.vector_store is not None:
            self.vector_store.close()


async def run_eval():

    evaluation_data = pd.read_csv(
        config["paths"]["evaluation_data_path"],encoding="latin1"
    )

    runner = RAGEvaluationRunner()

    try:

        for i in range(min(5, len(evaluation_data))):

            query = evaluation_data.iloc[i]["question"]
            human_answer = evaluation_data.iloc[i]["human_answer"]

            print("\n" + "=" * 70)
            print(f"Evaluation {i + 1}/5")
            print(f"Query: {query}, human_ans: {human_answer}")
            print("=" * 70)

            await runner.evaluate(
                query=query,
                human_answer=human_answer,
            )
            # exit()

    finally:

        runner.close()


if __name__ == "__main__":
    asyncio.run(run_eval())












# config=get_config()


# async def evaluate(query,ans):
    
#     # query="can you tell me total number of people working in Spacex"
#     # ans=""
    
#     vector_store,rag=create_rag_orchestrator()
#     try:
#         run_id,response=rag.invoke("123",query)
#         print(response)
#         llm=get_config()["llm"]
#         active_provider_name = llm["active_provider"]

#         # 2. Get the settings for that specific provider
#         provider_settings = llm["provider"][active_provider_name]

#         # 3. Initialize your config using the nested settings
#         llm_config = LLMConfig(
#             provider=active_provider_name,                 # "open_router"
#             model_name=provider_settings["model_name"],    # "nvidia/nemotron-3-super-120b-a12b:free"
#             temperature=provider_settings["temperature"]   # 0.5
#         )
#         llm_model = ChatModelProvider(llm_config).get_model()
            
#         result_store = RAGFileResultStore()
#         # run_id="run_f67d3b20bcb3403b96d142ce1df85b66"
#         payload = result_store.load(
#                 run_id
#             )
        
#         judge=create_LLM_judge()
    
#     # 6. Run the judge using the updated method signature
#         print("Evaluating chunks...")
#         verdicts = await judge.judge_chunks(
#             human_answer=ans,
#             payload=payload
#         )
#         RelevanceVerdictFileSave().save_file(run_id,verdicts)
        
#         # verdicts= RelevanceVerdictFileSave().load_file(run_id)
#         print("Evaluation Starting !!")
#         precision_metric=PrecisionAtK()
#         recall=RecallAtK()
#         mrr=MeanReciprocalRank()
#         ndcg=NDCGAtK()
#         hit_rate=HitRate()
#         faithfull=FaithfullnessEvaluationMetric(llm_model)
#         correctness=CorrectnessEvaluationMetric(llm_model)
#         relevance=AnswerRelevanceMetric(llm_model)
        
#         reterival_evaluator=RetrievalEvaluator([precision_metric,recall,mrr,ndcg,hit_rate])
#         reranker_evaluator=RerankerEvaluator([precision_metric,recall,mrr,ndcg,hit_rate])
#         generator_evaluator=GeneratorEvaluator([faithfull,correctness,relevance])
        
#         evaluator=RAGEvaluationPipeline(reterival_evaluator,reranker_evaluator,generator_evaluator)
#         result=await evaluator.run(payload,verdicts)
#         print(result)
#         EvaluationFileSave().save_file(run_id,result)
      
        
#     except Exception as e:
#         print("Exception", e)
#         traceback.print_exc()
#     finally:
#         vector_store.close()
    


# async def run_eval():
#     eval_data=pd.read_csv(config["paths"]["evaluation_data_path"])
#     for i in range(20):
#         query=eval_data["question"][i]
#         ans=eval_data["human_answer"][i]
#         await evaluate(query,ans)
        
# if __name__=="__main__":
#     asyncio.run(run_eval())
    