| Tech                     | Description                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------ |
| **LangChain**            | Framework for building applications with LLMs (used for chaining logic and memory).  |
| **Cohere API**           | Large Language Model (LLM) used for generating conversational medical responses.     |
| **Flask**                | Lightweight backend framework for serving the chatbot on the web.                    |
| **HTML/CSS + Bootstrap** | Frontend UI styled responsively with Bootstrap 5.                                    |
| **JavaScript**           | Handles real-time interaction and message rendering in the chat interface.           |
| **Pinecone (Vector DB)** | Stores vector embeddings for semantic search of medical documents and conversations. |
| **Python**               | Used for backend logic, API integration, and server-side processing.                 |


# 1. Clone the repo
git clone https://github.com/your-username/medibot-chat.git
cd medibot-chat

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Flask app
python app.py


# Folder Structure
medibot-chat/
│
├── static/
│   ├── styles.css
│
├── templates/
│   ├── index.html
│
├── app.py
├── chatbot.py  # LangChain + Cohere logic
├── vector_db.py  # Pinecone vector search logic
├── requirements.txt
└── README.md

 