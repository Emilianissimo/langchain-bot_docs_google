import sys
import dotenv
import openai
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

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

# Adding the history to our bot
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={'k': 16}),  # limit
    return_source_documents=True
)

chat_history = []

while True:
    query = input('Prompt (write exit to exit): ')

    if query == 'exit':
        print('Finishing')
        sys.exit()
    result = qa_chain({
        'question': query,
        'chat_history': chat_history
    })

    print('Answer: ' + result['answer'])
    chat_history.append((query, result['answer']))

