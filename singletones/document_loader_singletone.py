import os
from typing import List
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredExcelLoader


class DocumentLoaderSingletone:
    def __init__(self):
        self.documents = []
        self._load_documents()

    def _load_documents(self) -> None:
        for file in os.listdir('docs'):
            if file.endswith('.pdf'):
                pdf_path = './docs/' + file
                loader = PyPDFLoader(pdf_path)
                self.documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                doc_path = './docs/' + file
                loader = Docx2txtLoader(doc_path)
                self.documents.extend(loader.load())
            elif file.endswith('.txt'):
                text_path = './docs/' + file
                loader = TextLoader(text_path)
                self.documents.extend(loader.load())


documents: list = (DocumentLoaderSingletone()).documents

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
DOCUMENTS: List[Document] = text_splitter.split_documents(documents)
