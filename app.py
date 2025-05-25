from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf, text_splitter, download_HuggingFace
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import  create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY =  os.environ.get('PINECONE_API_KEY')
COHERE_API_KEY =  os.environ.get('COHERE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

embeddings = download_HuggingFace()

index_name = "aarc-medbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

chat = ChatCohere(temperature=0.4, max_tokens=200)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chat, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route('/')
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])

def chatbot():
    msg = request.form.get("msg", "")
    input = msg

    print(input)
    response = rag_chain.invoke({"input" : input})
    print(f"Response: ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)