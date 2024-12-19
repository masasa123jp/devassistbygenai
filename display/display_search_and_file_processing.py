# -*- coding: utf-8 -*-
import streamlit as st
from jira import JIRA
import openai
import os
import httpx
import json
from src.entities import Project, Issue

import pandas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import zipfile
import io
import docx
import PyPDF2
import mimetypes

from langchain import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain_community.document_loaders import PyPDFLoader,TextLoader,UnstructuredMarkdownLoader,UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.docstore.in_memory import InMemoryDocstore

import faiss
import tempfile

from PyPDF2 import PdfReader
from docx import Document
import pandas
import xlrd
import openpyxl

from display.display_home import display_home
from display.display_jira_assistance import display_jira_assistance

from src.error_handler import handle_api_errors
from src.jira_client import create_jira_client, get_all_projects, get_user_backlog
from src.openai_client import configure_openai, get_implementation_details
from src.config import *
from src.file_processing import *
from src.conversation import display_conversation
from src.export_import import *

#from transformers import GPT2Tokenizer
import uuid
import time


# インデックス情報のリスト
if os.path.exists(INDEX_INFO_PATH):
    with open(INDEX_INFO_PATH, 'r', encoding='utf-8') as f:
        index_list = json.load(f)
else:
    index_list = []

def initialize(embeddings:OpenAIEmbeddings):
    # ベクトルストアの初期化
    if os.path.exists(INDEX_FILE_PATH):
        vector_store = FAISS.load_local(
            INDEX_FILE_PATH,
            embeddings,
            allow_dangerous_deserialization=True  # 追加
            #allow_dangerous_serialization=True  # 追加
        )
    else:
        # 空のFAISSインデックスを初期化
        index = faiss.IndexFlatL2(OPENAI_CONFIG['dimension'])
        # 空のInMemoryDocstoreを使用
        docstore = InMemoryDocstore({})
        vector_store = FAISS(
            embedding_function=embeddings.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id={}
        )

# -------------------------
# 関数定義
# -------------------------

def is_file_processed(file_path):
    '''指定されたファイルが既に処理済みかどうかを確認する'''
    return any(entry['file_path'] == file_path for entry in index_list)

# ベクトルデータ追加用関数の修正
def process_file(file_path, text_splitter:RecursiveCharacterTextSplitter, vector_store:FAISS):
    ext = os.path.splitext(file_path)[1].lower()
    info_type = INFO_TYPES.get(ext, 'other')

    if is_file_processed(file_path):
        st.info(f'既に処理済みのファイルをスキップします: {file_path}')
        return

    content = ''
    if ext in ['.txt', '.py', '.md']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            st.warning(f'エンコーディングエラー: {file_path}')
            return
    elif ext == '.pdf':
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text
        except Exception as e:
            st.warning(f'PDFの読み込みに失敗しました: {file_path}')
            return
    elif ext == '.docx':
        try:
            doc = docx.Document(file_path)
            content = '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.warning(f'DOCXの読み込みに失敗しました: {file_path}')
            return
    elif ext in ['.xls', '.xlsx', '.xlsm']:
        try:
            content = read_excel_file(file_path)
        except Exception as e:
            st.warning(f'Excelファイルの読み込みに失敗しました: {file_path}')
            return
    else:
        st.info(f'サポートされていないファイルタイプをスキップします: {file_path}')
        return

    if not content.strip():
        st.info(f'空の内容をスキップします: {file_path}')
        return

    # コンテンツを分割してトークン数制限を超えないように処理
    chunks = text_splitter.split_text(content)

    total_tokens = sum(len(chunk) for chunk in chunks)

    # TOKEN_LIMITを超えた場合の処理
    if total_tokens > TOKEN_LIMIT:
        st.warning(f'ファイルがトークン数制限を超えています。分割して処理を行います。')

    # チャンクごとに埋め込み処理を実行
    for i, chunk in enumerate(chunks):
        unique_id = str(uuid.uuid4())
        vector_store.add_texts(
            [chunk],
            metadatas=[{'file_path': file_path, 'info_type': info_type}],
            ids=[unique_id]
        )

        # インデックス情報の追加
        index_entry = {
            'No': i + 1,
            'ids': unique_id,
            'file_path': file_path,
            'info_type': info_type,
            'chunks': len(chunks)
        }
        index_list.append(index_entry)

    st.success(f'ファイルを処理しました: {file_path}')

def process_zip(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                traverse_directory(tmpdirname)
    except Exception as e:
        st.warning(f'ZIPファイルの処理に失敗しました: {file_path}')

def traverse_directory(root_dir, text_splitter:RecursiveCharacterTextSplitter, vector_store:FAISS):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext == '.zip':
                process_zip(file_path)
            else:
                process_file(file_path, text_splitter, vector_store)

def save_index(vector_store:FAISS):
    # ベクトルストアの保存
    vector_store.save_local(
        INDEX_FILE_PATH,
        #allow_dangerous_deserialization=True,  # 追加
        #allow_dangerous_serialization=True  # 追加
    )

    # インデックス情報の保存
    with open(INDEX_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(index_list, f, ensure_ascii=False, indent=4)


# ファイルパスに基づいてベクトルデータを取得する関数
def get_vector_from_faiss(file_path, vector_store, index_list, embeddings:OpenAIEmbeddings):
    vectors = []
    #st.write(index_list)
#    try:
    for index in index_list:
        query_embedding = embeddings.embed_query(index['ids'])
        vdata = vector_store.similarity_search_by_vector(query_embedding, k=1)
        vectors.append(vdata[0].page_content)

    if not vectors:
        st.warning(f'{file_path} に対応するベクトルデータが見つかりませんでした。')
        return None
    return vectors
#    except Exception as e:
#        st.error(f'ベクトルデータの取得中にエラーが発生しました: {e}')
#        return None

# 取得したベクトルを使用してLLMに問い合わせる関数
def query_llm_with_vectors(vectors, prompt, llm_model, conversation):
#    try:
    # ベクトルデータを準備し、LLM用のプロンプトを作成
    vector_content = '\n'.join([str(vector) for vector in vectors])
    complete_prompt = f'以下のベクトルデータに基づいて、次のプロンプトに回答してください:\nベクトルデータ:\n{vector_content}\n\nプロンプト: {prompt}'
    #st.write(complete_prompt)

    # OpenAI LLMに問い合わせ
#    response = openai.ChatCompletion.create(
#        model=llm_model,
#        messages=[{'role': 'system', 'content': 'あなたは高度な情報処理システムです。'},
#                    {'role': 'user', 'content': complete_prompt}],
#        max_tokens=1000
#    )
    
    # get_implementation_detailsを使ってLLMに問い合わせ
    response = get_implementation_details(complete_prompt, openai.api_type, llm_model, conversation)
#    response = openai.chat.completions.create(
#        model=llm_model,
#        messages=[{'role': 'system', 'content': 'あなたは高度な情報処理システムです。'},
#                    {'role': 'user', 'content': complete_prompt}],
#        temperature=0.7
#    )
    #return response['choices'][0]['message']['content']
    return response.choices[0].message.content
#    except Exception as e:
#        st.error(f'LLMへの問い合わせ中にエラーが発生しました: {e}')
#        return None

# ベクトル取得とLLM問い合わせの処理を統合する関数
def vector_search_and_query_llm(prompt:str, embeddings:OpenAIEmbeddings, vector_store:FAISS, conversation):
#    try:
    # クエリをベクトルに変換
    #st.write(prompt)
    query_embedding = embeddings.embed_query(prompt)

    # 類似度検索（ベクトル検索）
    # FAISSベクトルストアをロード
    #vector_store = FAISS.load(INDEX_FILE_PATH, embeddings)
    results = vector_store.similarity_search_by_vector(query_embedding, k=1)  # 最上位1つのみ取得
    #st.write(results)
    file_path = ''
    file_content = ''
    answer = ''
    if results:
        # 最上位のドキュメント内容を取得
        doc = results[0]
        #st.write(doc)
        file_path = doc.metadata['file_path']

    #st.write(file_path)
    #st.write(len(index_list))
    index_list_by_file = []
    for index in index_list:
        if index['file_path'] == file_path:
            index_list_by_file.append(index)
    #st.write(len(index_list_by_file))
    # 指定されたファイルパスに基づいてベクトルを取得
    vectors = get_vector_from_faiss(file_path, vector_store, index_list_by_file, embeddings)

    if vectors:
        # プロンプトとベクトルを基にLLMに問い合わせ
        llm_response = query_llm_with_vectors(vectors, prompt, OPENAI_CONFIG['model_name_llm'], conversation)

        if not llm_response:
            st.error('LLMからの回答が得られませんでした。')
    else:
        st.error('指定されたファイルパスに基づくベクトルが見つかりませんでした。')
    return file_path, llm_response
#    except Exception as e:
#        st.error(f'ベクトル検索またはLLM問い合わせの処理中にエラーが発生しました: {e}')
#        return None, None

# Streamlit検索ページ
def search_query(user_query, embeddings, vector_store, conversation):
    with st.spinner('検索中...'):
        # ベクトル検索とLLMとの連携結果を取得
        top_result, llm_response = vector_search_and_query_llm(user_query, embeddings, vector_store, conversation)
        if top_result:
            # 検索結果の表示
            st.subheader('最上位のドキュメント')
            st.write(top_result)

            # LLMからの応答を表示
            st.subheader('LLMからの回答')
            st.write(llm_response)
        else:
            st.warning(llm_response)
        return llm_response

# -------------------------
# Streamlit UI
# -------------------------

#def count_tokens(text):
#    tokens = tokenizer.encode(text)
#    return len(tokens)

# PDFファイルのテキストを読み込んでトークン数を確認
#def count_tokens_in_file(file_path):
#    reader = PdfReader(file_path)
#    content = ''.join([page.extract_text() or '' for page in reader.pages])
#    return count_tokens(content)

################################################################################################

# -------------------------
# 検索ページ
# -------------------------
def display_search_and_file_processing(embeddings:OpenAIEmbeddings, text_splitter:RecursiveCharacterTextSplitter, vector_store:FAISS, conversation):
    st.header('情報検索')
    initialize(embeddings)

    question = st.text_area('プロンプトを入力してください RAG:', height=150)
    search_button = st.button('プロンプト送信')

    response = None
    if search_button and question:
        response = search_query(question, embeddings, vector_store, conversation)
    elif search_button:
        st.warning('プロンプトを入力してください。')

    st.header('ファイルのアップロードと処理')

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader('ファイルアップロード')
        uploaded_files = st.file_uploader(
            'ここにファイルをアップロードしてください（複数可）',
            type=['txt', 'py', 'pdf', 'docx', 'zip', 'md', 'xls', 'xlsx', 'xlsm'],
            accept_multiple_files=True
        )
        process_button = st.button('ファイルを処理する')

    with col2:
        st.subheader('処理の進捗')
        progress_bar = st.progress(0)
        status_text = st.empty()

    st.markdown('---')

    # フォルダーの処理
    st.header('指定フォルダ配下一括処理')
    folder_path = st.text_input(
        '処理するフォルダーのパスを入力してください:',
        value=INPUT_DIR_INDEX,
        help='指定されたフォルダー内のファイルも再帰的に処理されます。'
    )
    process_folder_button = st.button('フォルダーを処理する')

    if process_button and uploaded_files:
        with st.spinner('ファイルを処理中...'):
            total_files = len(uploaded_files)
            for idx, uploaded_file in enumerate(uploaded_files):
                # 一時ディレクトリにファイルを保存（拡張子を保持）
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                ext = os.path.splitext(uploaded_file.name)[1].lower()

                # 対応するファイル拡張子の処理
                if ext == '.zip':
                    process_zip(tmp_file_path)
                elif ext in ['.xls', '.xlsx', '.xlsm']:
                    content = read_excel_file(tmp_file_path)
                    st.write(f'**{uploaded_file.name}** の内容:\n{content}')
                else:
                    process_file(tmp_file_path, text_splitter, vector_store)

                # プログレスバーの更新
                progress = (idx + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f'処理中: {uploaded_file.name}')

            st.success('アップロードされた全てのファイルの処理が完了しました。')

    elif process_button:
        st.warning('処理するファイルをアップロードしてください。')

    if uploaded_files:
        st.session_state['uploaded_files_rag'] = read_uploaded_files(uploaded_files)  # Store uploaded files in session
        st.session_state['uploaded_files_size_rag'] = sum([file.size for file in uploaded_files])  # Calculate total size
        st.success(f'{len(uploaded_files)}ファイルがアップロードされました。')

    st.session_state['files_rag'] = st.session_state['uploaded_files_rag']
    st.session_state['files_size_rag'] = st.session_state['uploaded_files_size_rag']
    
    # アップロードファイルとフォルダファイルの合計を表示
    total_files = len(st.session_state['files_rag'])
    total_size = st.session_state.get('files_size_rag', 0) 
    st.write(f'・合計ファイル数: {total_files}')
    st.write(f'・合計ファイルサイズ: {total_size / 1024:.2f} KB')  # Show total size in KB

    # 各ファイルの内容を表示
    # uploaded_files_ragがsession_stateに存在するか確認
    if 'uploaded_files_rag' in st.session_state:
        st.write('・アップロードされたファイルの一覧:')
        uploaded_files = st.session_state['uploaded_files_rag']
        #st.write(uploaded_files)
        if uploaded_files:
            for key in uploaded_files.keys():
                # UploadedFileオブジェクトの属性にアクセス
                size = len(uploaded_files[key])
                #st.write(file)
                st.write(f'------->  {key} ({size / 1024:.2f} KB)')  # Display file size in KB
    else:
        st.write('uploaded_files_ragがセッションステートに存在しません。')

    if process_folder_button and folder_path:
        if os.path.isdir(folder_path):
            with st.spinner('フォルダーを処理中...'):
                traverse_directory(folder_path, text_splitter, vector_store)
                st.success(f"フォルダー '{folder_path}' の処理が完了しました。")
                save_index(vector_store)  # フォルダー処理が完了したらインデックスを保存する
        else:
            st.error(f'指定されたフォルダーが存在しません: {folder_path}')
    elif process_folder_button:
        st.warning('処理するフォルダーのパスを入力してください。')

        # プロンプト送信時のトークンサイズとコストを表示
        token_size = st.session_state.prompt_tokens[-1][0]
        cost = st.session_state.prompt_tokens[-1][1]
        st.sidebar.write(f'今回送信時のトークン: {token_size} ')
        st.sidebar.write(f'コスト: ${cost:.6f}')

    # 質問と回答の履歴を表示
    if 'rag_conversation' not in st.session_state:
        st.session_state.rag_conversation = []

    if response:
        st.session_state.rag_conversation.append({'question': question, 'answer': response})

    if st.session_state.rag_conversation:
        # 質問履歴のフィルタリング機能の追加
        filter_topic = st.text_input('フィルタリングしたいトピックを入力してくださいJIRA:')
        filtered_history = [entry for entry in st.session_state.rag_conversation if filter_topic.lower() in entry['question'].lower()]

        if filtered_history:
            st.write('フィルタリングされた質問履歴:')
            display_conversation(filtered_history, 'rag')
        else:
            st.write('フィルタリングされた結果はありません。')
        #display_conversation(st.session_state.rag_conversation, 'rag')
