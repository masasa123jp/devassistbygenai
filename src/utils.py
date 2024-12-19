# -*- coding: utf-8 -*-
import re

import openai
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain_openai import ChatOpenAI
#from langchain_openai.chat_models import ChatOpenAI
#from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
#from langchain.vectorstores.faiss import FAISS
from langchain_community.vectorstores import FAISS

def clean_text(text):
    text = text.replace('-\n', '')
    text = re.sub(r'\s+', ' ', text)
    return text

def split_text(text, wc_max):
    words = text.split()
    chunks = [' '.join(words[i:i + wc_max])
            for i in range(0, len(words), wc_max)]
    return chunks

def create_pdf_vector_db(pdf_url):
    # 1. PDFファイルの読み込みと分割
    pages = PyPDFLoader(pdf_url).load_and_split()
    
    # 2. OpenAIの埋め込みモデルを使ってテキストをベクトル化
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Latest OpenAI embeddings
    
    # 3. ベクトルストアの作成
    pdf_vector_store = FAISS.from_documents(pages, embeddings)
    
    # 4. 検索可能なリトリーバーオブジェクトを返す
    return pdf_vector_store.as_retriever(search_kwargs={"k": 5})

def create_chatbot(pdf_url, model):
    # ベクトルストアリトリーバーの作成
    retriever = create_pdf_vector_db(pdf_url)

    # Conversational retrieval chainの作成
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from flask import Blueprint, request, jsonify

# 1. バリデーション関数
def validate_request_data(data, required_fields):
    """リクエストデータのバリデーション"""
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    return True, None

# 2. エラーハンドリング関数
def handle_error(error_msg, status_code=400):
    """エラーメッセージを統一フォーマットで返却"""
    return jsonify({"error": {"message": error_msg, "code": status_code}}), status_code

# 3. データベース関数実行
def execute_function(func_name, params):
    """PostgreSQL関数を実行し、結果を返す"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(f"SELECT * FROM {func_name}({','.join(['%s'] * len(params))})", params)
        result = cur.fetchall() if cur.description else None
        conn.commit()
        return result, None
    except Exception as e:
        return None, str(e)
    finally:
        cur.close()
        conn.close()

def get_db_connection():
    """セキュアな方法でデータベース接続を確立"""
    try:
        connection = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432")
        )
        return connection
    except psycopg2.Error as e:
        raise RuntimeError(f"Database connection failed: {str(e)}")

def execute_query(query, params):
    """クエリを実行し結果を返す"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
    except psycopg2.Error as e:
        raise RuntimeError(f"Query execution failed: {str(e)}")

db_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT", "5432")
)

def get_db_connection():
    """接続プールから接続を取得"""
    try:
        return db_pool.getconn()
    except Exception as e:
        raise RuntimeError(f"Failed to get connection from pool: {str(e)}")

def release_db_connection(conn):
    """接続プールに接続を返す"""
    db_pool.putconn(conn)
