import openai
from dotenv import load_dotenv
import os

load_dotenv()

class ChatOpenAI:
    """
    This class pings open ai to create response for the list of messages
    """
    def __init__(self, model_name:str="gpt-3.5-turbo"):
        self.model_name = model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")
    
    def run(self, messages:list, text_only:bool=True):
        """
        Takes in list of messages and returns response
        """
        if not isinstance(messages, list):
            raise ValueError("Messages myst be a list")
        
        openai.api_key = self.openai_api_key
        response = openai.ChatCompletion.create(
            model=self.model_name, messages = messages
        )
        if text_only:
            return response.choices[0].message.content
        
        return response