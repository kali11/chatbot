import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from chat_service import ChatService
from dotenv import load_dotenv

import chainlit as cl

load_dotenv()
config = {
    "azure_search_service_name": os.environ["azure_search_service_name"],
    "azure_search_key": os.environ["azure_search_key"],
    "index_name": os.environ["index_name"],
    "azure_openai_api_base": os.environ["azure_openai_api_base"],
    "azure_openai_api_key": os.environ["azure_openai_api_key"],
    "azure_openai_api_version": os.environ["azure_openai_api_version"],
    "azure_openai_deployment_name": os.environ["azure_openai_deployment_name"],
}

chat_service = ChatService(config)

@cl.on_chat_start
async def on_chat_start():
    agent_executor = chat_service.create_agent()
    cl.user_session.set("runnable", agent_executor)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    res = runnable.invoke({"input": message.content})
    await cl.Message(content=res['output']).send()
