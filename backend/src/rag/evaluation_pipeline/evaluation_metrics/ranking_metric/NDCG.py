from typing import List
import math
from ....model.evaluation_context import EvaluatorContext
from ....evaluation_pipeline.evaluation_metrics.ranking_metric.metric import BaseRankingMetric

"It utilizes the 1-5 integer score from your LLM judge. It heavily rewards the system for putting a Score 5 (Perfect Match) at Rank 1, and penalizes it for pushing it down to Rank 4."


class NDCGAtK(BaseRankingMetric):
    def __init__(self,k:int=5) -> None:
        self.k=k
    
    
    @property
    def name(self):
        return f"NDCG@{self.k}"
    
    def calculate(self, chunks_id: List[str], context: EvaluatorContext) -> float:
        top_k=chunks_id[:self.k]
        if not top_k:
            return 0.0
        # 1. Calculate Actual DCG
        dcg = 0.0
        for rank, chunk in enumerate(top_k, start=1):
            # Fetch the actual 1-5 score from the context dictionary
            relevance_score = context.verdicts[chunk].score
            if relevance_score > 0:
                dcg += (math.pow(2, relevance_score) - 1) / math.log2(rank + 1)
                
            if dcg==0.0:
                return 0.0
            
        # 2. Calculate Ideal DCG (IDCG)
        # What is the maximum possible score if the best chunks were at the very top?
        all_possible_scores = sorted(
            [context.verdicts[c].score if c in context.verdicts else 0 for c in chunks_id],
            reverse=True
        )
        ideal_top_k_scores = all_possible_scores[:self.k]
        idcg = 0.0
        for rank, ideal_score in enumerate(ideal_top_k_scores, start=1):
            if ideal_score > 0:
                idcg += (math.pow(2, ideal_score) - 1) / math.log2(rank + 1)

        # 3. Normalize
        return dcg / idcg if idcg > 0 else 0.0