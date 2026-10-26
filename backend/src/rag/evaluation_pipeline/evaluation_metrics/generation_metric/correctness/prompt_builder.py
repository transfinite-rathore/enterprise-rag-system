from .....prompt_builder.base import IPromptBuilder
from .prompt import prompt
from .....model.rag_eval_context import RAGEvaluationPayload

class CorrectnessPromptBuilder(IPromptBuilder):
    
    def build_prompt(self, payload:RAGEvaluationPayload):
        query=payload.query
        expected_response=payload.expected_answer
        generated_answer=payload.response_result["answer"]
        print(f"query {query}, expected_response {expected_response}, generated_answer {generated_answer}")
        return prompt.format(query=query,human_answer=expected_response,generated_answer=generated_answer)