import chainlit as cl
from llmops.text_utils import TextFileLoader, CharacterTextSplitter
from llmops.vectordatabase import VectorDatabase
import asyncio
from llmops.retrieval_pipeline import RetrievalAugmentedQAPipeline, WandB_RetrievalAugmentedQAPipeline
from llmops.openai_utils.chatmodel import ChatOpenAI
import wandb
from llmops.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)

RAQA_PROMPT_TEMPLATE = """
Use the provided context to answer the user's query. 

You may not answer the user's query unless there is specific context in the following text.

If you do not know the answer, or cannot answer, please respond with "I don't know".

Context:
{context}
"""

raqa_prompt = SystemRolePrompt(RAQA_PROMPT_TEMPLATE)

USER_PROMPT_TEMPLATE = """
User Query:
{user_query}
"""

user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)

@cl.on_chat_start
async def on_chat_start():
    # files = None

    # # Wait for the user to upload a file
    # while files == None:
    #     files = await cl.AskFileMessage(
    #         content="Please upload a text file to begin!",
    #         accept=["text/plain"],
    #         max_size_mb=20,
    #         timeout=180,
    #     ).send()

    # file = files[0]

    msg = cl.Message(
        content=f"Loading Dataset ...", disable_human_feedback=True
    )
    await msg.send()
    # print(file.path)
    # print(file)
    text_loader = TextFileLoader('data/KingLear.txt')
    documents = text_loader.load_documents()
    # documents = [file.content]
    # print(documents)
    

    text_splitter = CharacterTextSplitter()
    split_documents = text_splitter.split_texts(documents)

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))
    chat_openai = ChatOpenAI()
    wandb.init(project="RAQA Example")
    raqa_retrieval_augmented_qa_pipeline = WandB_RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=chat_openai,
        wandb_project="RAQA from Scratch"
    )
    # Let the user know that the system is ready
    msg.content = f"Dataset loading is done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", raqa_retrieval_augmented_qa_pipeline)

@cl.on_message
async def main(message:str):
    chain = cl.user_session.get("chain")
    output = chain.run_pipeline(message,raqa_prompt, user_prompt)
    print(output)
    msg = cl.Message(content=f"{output}")
    # msg.prompt = output
    await msg.send()

