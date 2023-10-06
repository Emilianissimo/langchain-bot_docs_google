from typing import List
from langchain.schema import Document
from langchain.utilities import GoogleSearchAPIWrapper


class GoogleSearchiService:
    def __init__(self, engine: GoogleSearchAPIWrapper):
        self.max_results = 16
        self.search_engine = engine

    def search(self, query: str) -> List[Document]:
        internet_results = self.search_engine.run(query)[:self.max_results]
        return [Document(page_content=result) for result in internet_results]

