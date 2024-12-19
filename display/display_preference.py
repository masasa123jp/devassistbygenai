# -*- coding: utf-8 -*-
import streamlit as st
import os
import json

import pandas
import zipfile
import docx
import PyPDF2

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore

import faiss

from PyPDF2 import PdfReader
from docx import Document

from src.error_handler import handle_api_errors
from src.config import *
from src.file_processing import *

# インデックス情報のリスト
if os.path.exists(INDEX_INFO_PATH):
    with open(INDEX_INFO_PATH, 'r', encoding='utf-8') as f:
        index_list = json.load(f)
else:
    index_list = []

def save_index(vector_store, param_index_list):
    # ベクトルストアの保存
    vector_store.save_local(
        INDEX_FILE_PATH,
        #allow_dangerous_deserialization=True,  # 追加
        #allow_dangerous_serialization=True  # 追加
    )

    # インデックス情報の保存
    with open(INDEX_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(param_index_list, f, ensure_ascii=False, indent=4)

# -------------------------
# 設定ページ
# -------------------------
def display_preference(embeddings:OpenAIEmbeddings, text_splitter:RecursiveCharacterTextSplitter,vector_store, param_index_list):
    st.header('設定')

    st.subheader('インデックスの再構築')
    if st.button('インデックスを再構築する'):
        with st.spinner('インデックスを再構築中...'):
            try:
                # インデックスの再構築
                texts = []
                metadatas = []
                for entry in index_list:
                    file_path = entry['file_path']
                    # ファイルが存在するか確認
                    if os.path.exists(file_path):
                        ext = os.path.splitext(file_path)[1].lower()
                        try:
                            if ext in ['.txt', '.py', '.md']:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                            elif ext == '.pdf':
                                reader = PdfReader(file_path)
                                content = ''.join([page.extract_text() or '' for page in reader.pages])
                            elif ext in ['.xls', '.xlsx', '.xlsm']:
                                content = read_excel_file(file_path)  # Excelファイルの処理
                            else:
                                content = ''
                            if content.strip():
                                chunks = text_splitter.split_text(content)
                                texts.extend(chunks)
                                metadatas.extend([{'file_path': file_path, 'info_type': entry['info_type']} for _ in chunks])
                        except Exception as e:
                            st.warning(f'ファイルの読み込みに失敗しました: {file_path}')
                if texts:
                    # 新しいベクトルストアを作成
                    #dimension = 1536  # OpenAIの埋め込みベクトルの次元数
                    index = faiss.IndexFlatL2(OPENAI_CONFIG['dimension'],)
                    docstore = InMemoryDocstore({})
                    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
                    # インデックスの保存
                    save_index(vector_store, param_index_list)
                    st.success('インデックスの再構築が完了しました。')
                else:
                    st.info('再構築するテキストがありません。')
            except Exception as e:
                st.error(f'インデックスの再構築中にエラーが発生しました: {e}')

    st.subheader('インデックス情報')
    if index_list:
        st.write(f'総ファイル数: {len(index_list)}')
        for entry in index_list:
            st.write(f'- {entry['file_path']} ({entry['info_type']}) - チャンク数: {entry['chunks']}')
    else:
        st.write('インデックス情報が存在しません。')
