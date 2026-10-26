from .....model.evaluation_context import EvaluatorContext
from .....model.rag_eval_context import RAGEvaluationPayload
from .prompt_builder import CorrectnessPromptBuilder
from ..metric import BaseGenerationMetric
import json

class CorrectnessEvaluationMetric(BaseGenerationMetric):
    
    def __init__(self,llm_model) -> None:
        self.model=llm_model
        self.prompt_builder=CorrectnessPromptBuilder()
    
    @property
    def name(self) -> str:
        return "Correctness"
    
    async def compute(self, payload: RAGEvaluationPayload, context: EvaluatorContext) -> float:
        # return await super().compute(payload, context)
        # if not payload.expected_answer:
        #     print(f"payload")
        #     return 0.0 # Cannot compute correctness without a reference baseline
        
        prompt=self.prompt_builder.build_prompt(payload=payload)
        
        # FIX: Await the response, then extract the content
        response = await self.model.ainvoke(prompt)
        response_text = response.content
        
        try:
            if isinstance(response_text, list):
                response_text = "".join(
                    block.get("text", "") for block in response_text if isinstance(block, dict) and block.get("type") == "text"
                )

            # Strip markdown formatting just in case the LLM wrapped the JSON
            response_text = str(response_text).replace("```json", "").replace("```", "").strip()
            data = json.loads(response_text)
            
            # Normalize the 1-5 score to a 0.0 - 1.0 float for metric consistency
            score = float(data.get("score", 1))
            return (score-1)/4
            
        except Exception as e:
            # FIX: Added raw text output to debugging statement
            print(f"Correctness evaluation failed: {e}. Raw text: '{response_text}'")
            return 0.0