import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from .LLMFactory import LLMFactory
from ...model.llm_config import LLMConfig


load_dotenv()

class GeminiLLM(LLMFactory):
    def __init__(self,config:LLMConfig) -> None:
            self.config=config
        
    def create_model(self):
        model=ChatGoogleGenerativeAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=os.getenv("GEMINI_API_KEY") # pyright: ignore[reportArgumentType])
        )
        
        return model