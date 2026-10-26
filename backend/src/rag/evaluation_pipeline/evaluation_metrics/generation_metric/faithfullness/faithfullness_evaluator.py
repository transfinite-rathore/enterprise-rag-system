from .....model.evaluation_context import EvaluatorContext
from .....model.rag_eval_context import RAGEvaluationPayload
from .prompt_builder import FaithfullnessPromptBuilder
import json
from ..metric import BaseGenerationMetric

class FaithfullnessEvaluationMetric(BaseGenerationMetric):
    
    def __init__(self,llm_model) -> None:
        self.llm=llm_model
        self.prompt_builder=FaithfullnessPromptBuilder()
        
    @property
    def name(self) -> str:
        return "Faithfulness"
    
    async def compute(self, payload: RAGEvaluationPayload, context: EvaluatorContext) -> float:
        # print("faithfullness payload",payload)
        relevant_content=[chunk["content"] for chunk in payload.retrieved_result if chunk["chunk_id"] in context.relevant_ids]
        if len(relevant_content)==0:
            return 0.0 
        
        # FIX 1: Use \n instead of /n
        relevant_context="\n---\n".join(relevant_content)
        prompt=self.prompt_builder.build_prompt(context=relevant_context,payload=payload)
        
        # FIX 2: Extract the AIMessage object
        response = await self.llm.ainvoke(prompt)
        
        # FIX 3: Get the actual content out of the message
        response_text = response.content 
        
        try:
            if isinstance(response_text, list):
                response_text = "".join(
                    block.get("text", "") for block in response_text if isinstance(block, dict) and block.get("type") == "text"
                )

            # Strip markdown formatting just in case the LLM wrapped the JSON
            response_text = str(response_text).replace("```json", "").replace("```", "").strip()
            data = json.loads(response_text)
            
            # Note: Double check this normalization! 
            # If your prompt asks for a score of 0 or 1, this formula will yield negative numbers.
            # If your prompt asks for 1-5, this is correct.
            score = float(data.get("score", 0))
            # print(f"Faithfulnes score {score_out_of_5}, value {(score_out_of_5 - 1) / 4.0 }")
            return score  
            
        except Exception as e:
            # Print the raw text so you can debug what Gemini actually output
            print(f"Faithfulness evaluation failed: {e}. Raw text: '{response_text}'")
            return 0.0
        
        return response.content