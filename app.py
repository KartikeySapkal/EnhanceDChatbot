import os
import json
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import tempfile
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global dictionary to store conversation chains
conversation_chains = {}

load_dotenv()

def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    sec_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def serialize_chat_history(chat_history):
    return [
        {
            "role": "user" if isinstance(message, HumanMessage) else "bot",
            "content": message.content
        }
        for message in chat_history
    ]

def deserialize_chat_history(chat_history):
    return [
        HumanMessage(content=message["content"]) if message["role"] == "user" else AIMessage(content=message["content"])
        for message in chat_history
    ]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdfs' not in request.files:
            return redirect(request.url)

        pdf_files = request.files.getlist('pdfs')
        if not pdf_files:
            return redirect(request.url)

        temp_dir = tempfile.mkdtemp()
        pdf_paths = []
        for file in pdf_files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            pdf_paths.append(file_path)

        raw_text = get_pdf_text(pdf_paths)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore)

        session_id = str(uuid.uuid4())
        conversation_chains[session_id] = conversation_chain

        session['session_id'] = session_id
        session['chat_history'] = serialize_chat_history([])

        return redirect(url_for('chat'))

    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'session_id' not in session:
        return redirect(url_for('index'))

    session_id = session['session_id']
    conversation_chain = conversation_chains.get(session_id)
    chat_history = deserialize_chat_history(session.get('chat_history', []))

    if request.method == 'POST':
        user_question = request.form['question']
        response = conversation_chain({'question': user_question})
        chat_history = response['chat_history']

        session['chat_history'] = serialize_chat_history(chat_history)

    return render_template('chat.html', chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    if 'session_id' not in session:
        return redirect(url_for('index'))

    session_id = session['session_id']
    conversation_chain = conversation_chains.get(session_id)
    chat_history = deserialize_chat_history(session.get('chat_history', []))

    user_question = request.json.get('question')
    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']

    session['chat_history'] = serialize_chat_history(chat_history)

    return {"chat_history": serialize_chat_history(chat_history)}

if __name__ == '__main__':
    app.run(debug=True)
