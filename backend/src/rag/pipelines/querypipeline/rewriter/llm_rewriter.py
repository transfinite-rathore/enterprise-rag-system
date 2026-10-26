from ....model.user_query import UserQuery
from ....model.query_context import QueryContext
from .rewriter import QueryRewriter
from langchain_core.output_parsers import StrOutputParser
# from backend.src.rag.pipelines.querypipeline.rewriter.rewriter import QueryRewriter

class LLMQueryRewriter(QueryRewriter):
    
    def __init__(self,llm_model) -> None:
        self.llm=llm_model
    
    def rewrite_query(self, query: UserQuery) -> QueryContext:
        prompt = f"""
                Rewrite this query for document retrieval.
        
                Query:
                {query.query}
                Rules:
                - Preserve the original intent.
                - Add relevant keywords.
                - Do not answer the question.
                - Return only the rewritten query.

                """
        chain = self.llm | StrOutputParser()
        
        # Invoke the chain instead of just the LLM
        response_text = chain.invoke(prompt)
        
        processed_query = QueryContext(
            user_query=query.query,
            processed_query=response_text 
        )
        
        return processed_query
        # response=self.llm.invoke(prompt)
        # processed_query=QueryContext(
        #             user_query=query.query,
        #             processed_query=response.content )
        
        # return processed_query