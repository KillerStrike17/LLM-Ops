import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable
from llmops.openai_utils.embedding import EmbeddingModel
import asyncio

def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

class VectorDatabase:
    def __init__(self, embedding_model:EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key:str, vector:np.array)->None:
        """
        Adding elements to the dictionary vectors, with key as key and value as vector
        """
        self.vectors[key] = vector

    def search(self, query_vector:np.array,k:int, distance_measure:Callable = cosine_similarity)->List[Tuple[str, float]]:
        """
        calculates cosine similarity between query vector and vector in the database and then sort the result and 
        returns the top k values by slicing the list
        """
        scores = [
            (key, distance_measure(query_vector, vector)) for key, vector in self.vectors.items()
        ]
        return sorted(scores, key = lambda x:x[1], reverse = True)[:k]
    
    def search_by_text(self, query_text:str, k:int, distance_measure:Callable = cosine_similarity, return_as_text:bool = False) -> List[Tuple[str, float]]:
        """
        This function converts the text query to embeddings and then calls the seach function
        """
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        """
        This function returns the value of the parameter key in the vector dictionary
        """
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        """
        create a database from a list of texts. text is key where as embedding is the mapping
        """
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self