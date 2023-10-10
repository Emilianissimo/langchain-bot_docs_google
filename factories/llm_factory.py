from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from singletones.document_loader_singletone import DOCUMENTS
from singletones.settings_singletone import SettingsSingleTone
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain


class LLMFactory:
    def __init__(self):
        # Initialize settings
        self._settings = SettingsSingleTone(provider='azure')
        # Setting memory
        self.memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
        # LLM
        self.llm = self._setup_llm()
        # Embeddings
        self._embeddings = self._setup_embeddings()
        # Vector DB with persistent folder
        self._vectordb = self._setup_vectordb()
        # self._vectordb.persist()
        # Google search
        self.google_search_engine = self._setup_google_search_api_wrapper()
        # Question-Answer chain + history of it
        self.chain = self._setup_chain()

    def _setup_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            temperature=0.5,
            openai_api_key=self._settings.OPENAI_API_KEY,
            model_kwargs={'engine': self._settings.DEPLOYMENT_NAME}
        )

    def _setup_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            openai_api_key=self._settings.OPENAI_API_KEY,
            model_kwargs={'engine': self._settings.EMBEDDED_MODEL_NAME}
        )

    def _setup_vectordb(self) -> Chroma:
        return Chroma.from_documents(
            DOCUMENTS,
            embedding=self._embeddings,
            persist_directory='./data'
        )

    def _setup_google_search_api_wrapper(self) -> GoogleSearchAPIWrapper:
        return GoogleSearchAPIWrapper(
            google_cse_id=self._settings.GOOGLE_CSE_ID,
            google_api_key=self._settings.GOOGLE_API_KEY,
        )

    def _setup_chain(self):
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self._vectordb.as_retriever(search_kwargs={'k': 12}),
            return_source_documents=False,  # Agent is not working with those keys into the response
            verbose=True,
            memory=self.memory,
            max_tokens_limit=4000
        )
