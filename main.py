import sys
import openai
from typing import List
from langchain.schema import Document
from factories.llm_factory import LLMFactory
from constants import YELLOW, GREEN, WHITE, RED
from singletones.document_loader_singletone import DOCUMENTS
from services.google_search_service import GoogleSearchiService


# Init all required components
llm: LLMFactory = LLMFactory()

# Init google service
google_search_engine: GoogleSearchiService = GoogleSearchiService(engine=llm.google_search_engine)

chat_history = []

print(f"{YELLOW}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents. Ask bot about data included '
      'or else.')
print('---------------------------------------------------------------------------------')

while True:
    query: str = input(f'{GREEN}Prompt (write exit to end or press F): ')

    if query in ['exit', 'F', 'f']:
        print('Finishing')
        sys.exit()

    if query == '':
        continue

    internet_results: List[Document] = google_search_engine.search(query)

    try:
        result: dict = llm.chain({
            'question': query,
            'chat_history': chat_history,
            'documents': DOCUMENTS + internet_results
        })
        print(f"{WHITE}Answer: " + result["answer"])
        chat_history.append((query, result['answer']))
    except openai.error.InvalidRequestError:
        print(f'{RED}Your context length is more than model able to handle. Please, delete DATA folder and rerun it')
        print(f'{RED}Finishing')
        sys.exit()
