from langchain_chroma import Chroma
# from langchain_together import TogetherEmbeddings
from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from util import check_exist
import time



class VDB:
    def __init__(self, name, pdf_path, db_path = "./DB", pdf_mode = "page", chunk_size = 1000, chunk_overlap = 100):
        self.db_name = name
        self.pdf_path = pdf_path
        self.db_path = db_path + "/" + self.db_name
        
        # self.embedding =  TogetherEmbeddings(
        #     model="togethercomputer/m2-bert-80M-32k-retrieval",
        #     api_key="84e8df9a595039765758ae96105665d37e873e9619a2c209ee31a108db5875ef"
        # )

        self.embedding = GoogleGenerativeAIEmbeddings(
            google_api_key="AIzaSyCuwAN1ZJaGUUUyJKemHFmW_EzJQszYnxE",
            model="models/embedding-001",
        )
        
        
        self.initDB(pdf_mode, chunk_size, chunk_overlap)
        self.retriever = self.vector_store.as_retriever()
    
    def initDB(self, pdf_mode, chunk_size, chunk_overlap):
        print(f"Locating DB: {self.db_path}")
        if not check_exist(self.db_path):
            print("Creating New Vector DB")
            self.create_db()
            self.add_embeddings(pdf_mode, chunk_size, chunk_overlap)
        else: 
            print("Loading DB..")
            self.create_db()
    
    def create_db(self):
        self.vector_store = Chroma(
            collection_name=self.db_name,
            embedding_function=self.embedding,
            persist_directory=self.db_path    
        )

    def add_embeddings(self, pdf_mode, chunk_size, chunk_overlap):
        loader = PyPDFLoader(self.pdf_path, mode= pdf_mode)
        print(f"Loading PDF: {self.pdf_path}")
        start_time = time.time()
        docs = loader.load()
        print(f"Pdf loaded in {time.time() - start_time}")
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        # split_docs = text_splitter.split_documents(docs)
        # self.vector_store.add_documents(split_docs)
        start_time = time.time()
        print(f"Loading data into VDB")
        self.vector_store.add_documents(docs)
        print(f"Data loaded into VDB in {time.time() - start_time}")

    def similarity_search(self, query, k=1):
        return self.vector_store.similarity_search(query, k)
    
    def similarity_search_with_score(self, query, k=1):
        return self.vector_store.similarity_search_with_score(query, k)
    
    def similarity_search_with_vector(self, query, k=1):
        return self.vector_store.similarity_search_by_vector(embedding=self.embedding.embed_query(query), k=k)
    
    def get_db(self):
        return self.vector_store
