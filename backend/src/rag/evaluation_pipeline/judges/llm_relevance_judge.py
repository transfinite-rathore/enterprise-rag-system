from typing import Dict, List,Optional
import asyncio
from pydantic import ValidationError
import json
from src.rag.model.chunk_relevancy import ChunkRelevantVerdict
from src.rag.model.retrieved_document import RetrievedDocument
from src.rag.model.rag_eval_context import RAGEvaluationPayload
from .base import RelevanceJudge
from  .prompt_builder import PromptBuilder


class LLMRelevanceJudge(RelevanceJudge):
    def __init__(self,llm_model,batch_size:int=16) -> None:
        self.llm=llm_model
        self.batch_size=batch_size
        self.prompt_builder=PromptBuilder()
    
    async def _judge_chunk_batch(self, chunks: List[Dict[str,str]], query: str, human_response:str) -> List[ChunkRelevantVerdict]:
        
        # for 
        prompt=self.prompt_builder.build_prompt(query,human_response,chunks)
        
        response_message= await self.llm.ainvoke(prompt)
        response_text = response_message.content
        
        verdicts = []
        
        # 3. Parse and store in Pydantic models
        try:
            if isinstance(response_text, list):
                response_text = "".join(
                    block.get("text", "") for block in response_text if isinstance(block, dict) and block.get("type") == "text"
                )
            elif not isinstance(response_text, str):
                response_text = str(response_text)
                
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            parsed_data = json.loads(response_text)
            
            for item in parsed_data:
                try:
                    # Map the LLM's 'relevant' key to your model's 'is_relevant' field
                    verdict = ChunkRelevantVerdict(
                        chunk_id=item["chunk_id"],
                        is_relevant=item.get("relevant", False),
                        reason=item.get("reason", "No reason provided by LLM"),
                        score=item.get("score", 1)
                    )
                    verdicts.append(verdict)
                    
                except ValidationError as validation_err:
                    print(f"Skipping chunk {item.get('chunk_id')} due to validation error: {validation_err}")
                    # Create a default failure verdict so the pipeline doesn't break
                    verdicts.append(ChunkRelevantVerdict(
                        chunk_id=item.get("chunk_id", "unknown_id"),
                        is_relevant=False,
                        reason="Failed Pydantic validation",
                        score=0
                    ))
                    
            return verdicts
            
        except json.JSONDecodeError as decode_err:
            print(f"Failed to parse LLM response as JSON: {decode_err}")
            print(f"Raw response: {response_text}")
            
            # Fallback: if the entire batch fails to parse, return negative verdicts for all chunks in this batch
            return [
                ChunkRelevantVerdict(
                    chunk_id=chunk_id,
                    is_relevant=False,
                    reason="LLM returned invalid JSON format",
                    score=0
                )
                for chunk_id in chunks.keys()
            ]
            
    
      
        
        
    async def judge_chunks(self, human_answer: Optional[str], payload: RAGEvaluationPayload) -> Dict[str, ChunkRelevantVerdict]:
        
        query = payload.query
        
        # Merge and deduplicate chunks from both retrieval and reranking stages
        all_chunks_map = {c["chunk_id"]: c for c in payload.retrieved_result}
        for c in payload.reranked_result:
            all_chunks_map[c["chunk_id"]] = c
            
        unique_chunks = list(all_chunks_map.values())
        
        # Batch the deduplicated chunks
        batches = [
            {chunk["chunk_id"]: chunk["content"] for chunk in unique_chunks[i : i + self.batch_size]}
            for i in range(0, len(unique_chunks), self.batch_size)
        ]
        
        # Ensure _judge_chunk_batch is defined as: def _judge_chunk_batch(self, batch, query, human_answer)
        tasks = [
            self._judge_chunk_batch(batch, query, human_answer)
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*tasks)
        flattened_verdicts = [verdict for batch_list in batch_results for verdict in batch_list]
        
        return {v.chunk_id: v for v in flattened_verdicts}
