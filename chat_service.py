from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.retrievers import AzureCognitiveSearchRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

class ChatService:
    def __init__(self, config) -> None:
        self.config = config

    def create_agent(self):
        llm = AzureChatOpenAI(
            azure_deployment=self.config["azure_openai_deployment_name"], 
            azure_endpoint=self.config["azure_openai_api_base"],
            api_key=self.config["azure_openai_api_key"],
            temperature=0,
            api_version=self.config["azure_openai_api_version"]
        )

        retriever = AzureCognitiveSearchRetriever(
            service_name=self.config["azure_search_service_name"],
            api_key=self.config["azure_search_key"],
            index_name=self.config["index_name"],
            content_key="content", 
            top_k=5
        )

        retriever_tool = create_retriever_tool(
            retriever,
            "search_legal_code",
            "Searches and returns legal regulations in Norway",
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant.""",
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
            ]
        )

        chat_history = ChatMessageHistory()

        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            memory_key='chat_history',
            return_messages=True,
            input_key='input',
            output_key='output'
        )

        agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool], memory=memory, verbose=True)
        return agent_executor
