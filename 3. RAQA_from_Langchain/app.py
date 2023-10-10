import chainlit as cl
import asyncio
import wandb
import pandas as pd
import pinecone
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
import langchain
from langchain.cache import InMemoryCache

pinecone.init(
    api_key= os.environ['PINECONE_API_KEY'],
    environment= os.environ['PINECONE_ENV']
)

index_name = 'movie-review-index'
index = pinecone.Index(index_name)


@cl.on_chat_start
async def on_chat_start():

    msg = cl.Message(
        content=f"Loading Dataset ...", disable_human_feedback=True
    )
    await msg.send()
    
    text_field = "text"

    store = LocalFileStore("./cache/")

    core_embeddings_model = OpenAIEmbeddings()
    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model,
        store,
        namespace= core_embeddings_model.model
    )
    vectorstore = Pinecone(
        index, embedder.embed_query, text_field
    )
    # docsearch = Pinecone.from_existing_index(
    #     index_name=index_name, embedding=embedder.embed_query, namespace=None
    # )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever()
    handler = StdOutCallbackHandler()

    qa_with_sources_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        callbacks=[handler],
        return_source_documents=True
    )
    langchain.llm_cache = InMemoryCache()
    # Let the user know that the system is ready
    msg.content = f"Dataset loading is done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", qa_with_sources_chain)

@cl.on_message
async def main(message:str):
    chain = cl.user_session.get("chain")
    output = chain({"query":message})
    # print(output)
    msg = cl.Message(content=f"{output['result']}")
    # msg.prompt = output
    await msg.send()