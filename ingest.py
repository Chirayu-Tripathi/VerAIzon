from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# Create vector database
class VectorStore:
    """Class to create VectorDB"""
    def __init__(self, data_path, db_path):
        """
        Initialize the variables for VectorStore.
        """
        self.data_path = data_path
        self.db_path = db_path
        
    def create_vector_db(self):
        """
        function to build vector DB.
        """
        DATA_PATH = self.data_path
        DB_FAISS_PATH = self.db_path
        loader_pdf = DirectoryLoader(DATA_PATH,
                                 glob=f'*.pdf',
                                 loader_cls=PyPDFLoader)

        loader_text = DirectoryLoader(DATA_PATH,
                                 glob=f'*.txt',
                                 loader_cls=TextLoader)

        documents_pdf = loader_pdf.load()
        documents_text = loader_text.load()
        documents = documents_pdf + documents_text
        for i in documents:
            i.page_content = i.page_content.replace(' . ','').replace('\n',' ')
            i.page_content = re.sub(r'\.+', ".", i.page_content)
            i.page_content = ' '.join(i.page_content.split())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                       chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2',
                                           model_kwargs={'device': 'cuda'})

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    data_path = '/home/chirayu.tripathi/hackathon/'
    db_path = '/home/chirayu.tripathi/hackathon/vectorstore/db_faiss'
    obj = VectorStore(data_path, db_path)
    obj.create_vector_db()

