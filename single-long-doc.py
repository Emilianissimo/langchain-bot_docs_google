import dotenv
import openai
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

config = dotenv.dotenv_values('.env')

openai.api_key = config.get('OPENAI_API_KEY')
openai.api_base = config.get('OPENAI_BASE')
openai.api_type = config.get('OPENAI_TYPE')
openai.api_version = config.get('OPENAI_VERSION')


pdfLoader = PyPDFLoader('docs/cv.pdf')
documents = pdfLoader.load()

# LLM

llm = ChatOpenAI(
    engine=config.get('DEPLOYMENT_NAME'),
    openai_api_key=config.get('OPENAI_API_KEY')
)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings(engine=config.get('EMBEDDED_MODEL_NAME'), openai_api_key=config.get('OPENAI_API_KEY'))

vectordb = Chroma.from_documents(
    documents,
    embedding=embedding,
    persist_directory='./data'
)
vectordb.persist()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={'k': 16}),  # limit amount of chunks to send
    return_source_documents=True
)

result = qa_chain({'query': 'Who is the CV about?'})
print(result['result'])
