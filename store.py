from src.helper import load_pdf, text_splitter, download_HuggingFace
from pinecone import  Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os 

load_dotenv()

PINECONE_API_KEY =  os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_docs = load_pdf(data="Data/")
text_chunks = text_splitter(extracted_docs)
embeddings = download_HuggingFace()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "aarc-medbot"

pc.create_index(
    
        name=index_name,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)