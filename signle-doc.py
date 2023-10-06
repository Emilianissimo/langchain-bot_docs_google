import dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

config = dotenv.dotenv_values('.env')

openai.api_key = config.get('OPENAI_API_KEY')
openai.api_base = config.get('OPENAI_BASE')
openai.api_type = config.get('OPENAI_TYPE')
openai.api_version = config.get('OPENAI_VERSION')


pdfLoader = PyPDFLoader('docs/cv.pdf')
pages = pdfLoader.load()

# LLM

llm = ChatOpenAI(
    engine=config.get('DEPLOYMENT_NAME'),
    openai_api_key=config.get('OPENAI_API_KEY')
)

chain = load_qa_chain(llm=llm)
query = 'Please, tell me the context of this document'

try:
    response = chain.run(input_documents=pages, question=query)
except openai.error.InvalidRequestError:
    response = chain.run(input_documents=pages[:5], question=query)

print(response)

