from dotenv import load_dotenv
from openai.embeddings_utils import get_embeddings, aget_embeddings, get_embedding, aget_embedding
import openai
from typing import List
import os
import asyncio

class EmbeddingModel:
    """
    This class contains functionalities to generate embeddings from the 
    list of texts or text asynchronously or in sync.
    """
    def __init__(self, embeddings_model_name:str = "text-embedding-ada-002"):
        """
        Loads the OpenAI Api key and sets the embedding model
        """
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY environment variables is not set. Please set it to your openAI API key")
        
        openai.api_key = self.openai_api_key
        self.embeddings_model_name = embeddings_model_name
    
    async def async_get_embeddings(self, list_of_text:List[str])->List[List[float]]:
        """
        This function takes in a list of strings and uses openai api 
        aget_embeddings to get the list of embeddings back. The process is asynchronous in nature

        """
        return await aget_embeddings(
            list_of_text = list_of_text, engine = self.embeddings_model_name
        )
    
    async def async_get_embedding(self, text: str) -> List[float]:
        """
        This function takes in a string and uses openai api 
        aget_embedding to get the list of embeddings back. The process is asynchronous in nature

        """
        return await aget_embedding(text=text, engine=self.embeddings_model_name)
    
    def get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        """
        This function takes in a list of strings and uses openai api 
        get_embeddings to get the list of embeddings back. The process is synchronous in nature

        """
        return get_embeddings(
            list_of_text=list_of_text, engine=self.embeddings_model_name
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        This function takes in a string and uses openai api 
        get_embedding to get the list of embeddings back. The process is synchronous in nature

        """
        return get_embedding(text=text, engine=self.embeddings_model_name)

if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    print(embedding_model.get_embedding("Hello, world!"))
    print(embedding_model.get_embeddings(["Hello, world!", "Goodbye, world!"]))
    print(asyncio.run(embedding_model.async_get_embedding("Hello, world!")))
    print(
        asyncio.run(
            embedding_model.async_get_embeddings(["Hello, world!", "Goodbye, world!"])
        )
    )