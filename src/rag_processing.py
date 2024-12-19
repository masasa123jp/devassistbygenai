# -*- coding: utf-8 -*-
import os
import json
import tempfile
import zipfile
import faiss
import openai
from langchain import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
from src.error_handler import handle_api_errors
from src.config import OPENAI_CONFIG

# OpenAI APIの設定
#OPENAI_CONFIG = {
#    "api_base": 'https://api.openai.com/v1',
#    "api_key": os.environ.get('OPENAI_API_KEY'),
#    "model_name_llm": "gpt-4o-mini",
#    "model_name_embeddings": "text-embedding-3-small",
#    "dimension": 1536,  # OpenAIの埋め込みベクトルの次元数
#}

# 定数の設定
TOKEN_LIMIT = 1000000
INDEX_FILE_PATH = 'path/to/faiss_index'  # 適切なファイルパスに変更してください
INDEX_INFO_PATH = 'path/to/index.json'  # 適切なファイルパスに変更してください

embeddings = OpenAIEmbeddings(model=OPENAI_CONFIG["model_name_embeddings"])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=100, length_function=len)

# ベクトルストアの初期化
def initialize_vector_store():
    # ベクトルストアのロード
    if os.path.exists(INDEX_FILE_PATH):
        vector_store = FAISS.load_local(INDEX_FILE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store, True
    else:
        return None, False

# アップロードファイルの処理
@handle_api_errors
def process_uploaded_files(uploaded_files):
    data = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == 'application/pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                loader = PyPDFLoader(tmp_file.name)
                data.extend(loader.load_and_split(text_splitter))
        elif uploaded_file.type == 'text/plain':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                loader = TextLoader(tmp_file.name)
                data.extend(loader.load_and_split(text_splitter))
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                loader = UnstructuredWordDocumentLoader(tmp_file.name)
                data.extend(loader.load_and_split(text_splitter))
        # 他のファイルタイプも必要に応じて追加
    return data

# LLMに問い合わせる関数
def query_llm_with_vectors(vectors, prompt, llm_model):
    vector_content = '\n'.join([str(vector) for vector in vectors])
    complete_prompt = f"以下のベクトルデータに基づいて、次のプロンプトに回答してください:\nベクトルデータ:\n{vector_content}\n\nプロンプト: {prompt}"

    response = openai.chat.completions.create(
        model=llm_model,
        messages=[{"role": "system", "content": "あなたは高度な情報処理システムです。"},
                  {"role": "user", "content": complete_prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# ベクトルデータの処理
def process_vector_data(file_path):
    if os.path.exists(file_path):
        # ここでファイルを処理し、ベクトルを生成
        # 具体的な処理を実装
        pass

# RAG関連のメイン処理関数
def run_rag_process(uploaded_files):
    vector_store, rag_data_available = initialize_vector_store()

    if not rag_data_available:
        return "RAGデータが利用できません。"

    data = process_uploaded_files(uploaded_files)

    # ベクトルストアを作成
    if data:
        vectors = FAISS.from_documents(data, embeddings)
        # Conversational Retrieval Chainを作成
        conversation = ConversationalRetrievalChain.from_llm(llm=OpenAI(temperature=0, model_name=OPENAI_CONFIG["model_name_llm"]), retriever=vectors.as_retriever())
        return conversation
    return "データが処理されていません。"
