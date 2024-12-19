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
from display.display_search_and_file_processing import display_search_and_file_processing
from display.display_preference import display_preference, save_index
from display.display_dynamic_qa1 import display_dynamic_qa1

from src.error_handler import handle_api_errors
from src.jira_client import create_jira_client, get_all_projects, get_user_backlog
from src.openai_client import configure_openai, get_implementation_details
from src.config import *
from src.file_processing import *
from src.conversation import display_conversation
from src.export_import import *
from display.display_dynamic_qa2 import display_dynamic_qa2, query_llm_with_vectors

#from transformers import GPT2Tokenizer
import uuid
import time

# Streamlitのページ設定
st.set_page_config(page_title="開発支援ツール", layout="wide")

# セッションでのトークン数をトラッキング
if 'prompt_tokens' not in st.session_state:
    st.session_state.prompt_tokens = []
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0

# セッション状態の初期化
INITIAL_KEYS = [
    'home_conversation', 'project_conversation', 'rag_conversation',
    'user_account_id', 'projects', 'selected_project', 'project_prompt',
    'backlog_issues', 'selected_issue', 'selected_issue_key',
    'uploaded_files_home', 'folder_files_home', 'uploaded_files_rag', 'folder_files_rag', 'files_rag',
]

for key in INITIAL_KEYS:
    if key not in st.session_state:
        st.session_state[key] = [] if 'home_conversation' or 'project_conversation'or 'rag_conversation' \
                                    or 'projects' or 'selected_project' or 'backlog_issues' \
                                    or 'uploaded_files_home' or 'folder_files_home' \
                                    or 'uploaded_files_rag' or 'folder_files_rag' or 'files_rag' \
                                    in key else None

# チェックボックスの状態を初期化
for checkbox in ['home_checkbox_state', 'home_checkbox_state2', 'home_checkbox_state3', 'rag_checkbox_state', 'rag_checkbox_state2']:
    if checkbox not in st.session_state:
        st.session_state[checkbox] = False

# 読み込んだファイルのサイズ情報を初期化
for size in ['uploaded_files_size_rag', 'folder_files_size_rag', 'files_size_rag']:
    if size not in st.session_state:
        st.session_state[size] = 0


# ナビゲーションメニュー
def display_navigation_menu():
    st.sidebar.title("開発支援 by Gen-AI \n\n ナビゲーション")
    # 定数の追加
    return st.sidebar.radio("選択", list(PAGES.values()))

data = read_py_files_to_string(os.path.dirname(__file__))

embeddings = OpenAIEmbeddings(model=OPENAI_CONFIG["model_name_embeddings"])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=100, length_function=len)

# LLMの初期化
llm = OpenAI(temperature=0, model_name=OPENAI_CONFIG["model_name_llm"])

# ベクトルストアのロードとRAGデータの有無の判定
try:
    if os.path.exists(INDEX_FILE_PATH):
        # ベクトルストアをロード
        vectorstore = FAISS.load_local(INDEX_FILE_PATH, embeddings, allow_dangerous_deserialization=True)
        rag_data_available = True
    else:
        vectorstore = None
        rag_data_available = False
except FileNotFoundError:
    # ベクトルストアが存在しない場合
    vectorstore = None
    rag_data_available = False

memory = ConversationBufferMemory(memory_key="history", return_messages=True)
# Conversational Retrieval ChainまたはConversation Chainの作成
if rag_data_available:
    # リトリーバーの設定
    retriever = vectorstore.as_retriever()
    # Conversational Retrieval Chainを作成
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )
else:
    # Conversation Chainを作成
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        verbose=False
    )

def main():
    # OpenAI APIの設定
    configure_openai()

    # メインの処理を実行
    page = display_navigation_menu()

    if page == "ホーム":
        display_home(data, conversation)
    elif page == "開発支援(JIRA連携)":
        display_jira_assistance(conversation)
    elif page == "ダイナミックQ&A":
        display_dynamic_qa2(conversation)
    #elif page == "RAG実験":
    #    display_rag_experiment()
    #elif page == "ファイル処理":
    #    display_file_process()
    elif page == "情報検索 and ファイル処理":
        display_search_and_file_processing(embeddings, text_splitter, vector_store, conversation)
    elif page == "質問履歴":
        display_question_history()
    elif page == "設定":
        display_preference(embeddings, text_splitter,vector_store, index_list)
    elif page == "SessionData":
        display_session_data()

    # これまでに送ったプロンプト送信時のトークンサイズと総計コストを表示
    display_total_tokens_and_cost()

    # セッションサイズ表示
    display_session_size()

    # セッションクリアボタンの追加
    if st.sidebar.button("セッションをクリア"):
        clear_session()  # セッションをクリアする関数を呼び出す
        st.sidebar.success("セッションがクリアされました。")

# RAG実験
def display_rag_experiment():
    st.title(PAGES["RAG"])

    # チェックボックスの定義
    prompt_program = "このアプリの使い方を教えてください。RAG(Retrieval-Augmented Generation、検索拡張生成)"
    selected_checkbox = st.checkbox(label=prompt_program, value=st.session_state.rag_checkbox_state)

    question = ""
    # プロンプトを入力/作成
    if not selected_checkbox:
        user_prompt = st.text_area("プロンプトを入力してください RAG: ", height=200)
        question = user_prompt
    else:
        user_prompt = prompt_program + "ただし、認証情報の文字列は伏せてください。"
        question = prompt_program

    # アップロードファイルの処理
    uploaded_files = st.file_uploader("ファイルをアップロードしてください (最大4ファイル) RAG", type=["txt", "pdf", "docx", "xls", "xlsx", "xlsm", "py", "zip", "md"], accept_multiple_files=True)
    # アップロードされたファイルの処理を関数を使って実行

    if uploaded_files:
        st.session_state['uploaded_files_rag'] = read_uploaded_files(uploaded_files)  # Store uploaded files in session
        st.session_state['uploaded_files_size_rag'] = sum([file.size for file in uploaded_files])  # Calculate total size
        st.success(f"{len(uploaded_files)}ファイルがアップロードされました。")

    st.session_state['files_rag'] = st.session_state['uploaded_files_rag']
    st.session_state['files_size_rag'] = st.session_state['uploaded_files_size_rag']
    
    # アップロードファイルとフォルダファイルの合計を表示
    total_files = len(st.session_state['files_rag'])
    total_size = st.session_state.get('files_size_rag', 0) 
    st.write(f"・合計ファイル数: {total_files}")
    st.write(f"・合計ファイルサイズ: {total_size / 1024:.2f} KB")  # Show total size in KB

    # 各ファイルの内容を表示
    # uploaded_files_ragがsession_stateに存在するか確認
    if 'uploaded_files_rag' in st.session_state:
        st.write("・アップロードされたファイルの一覧:")
        uploaded_files = st.session_state['uploaded_files_rag']
        #st.write(uploaded_files)
        if uploaded_files:
            for key in uploaded_files.keys():
                # UploadedFileオブジェクトの属性にアクセス
                size = len(uploaded_files[key])
                #st.write(file)
                st.write(f"------->  {key} ({size / 1024:.2f} KB)")  # Display file size in KB
    else:
        st.write("uploaded_files_ragがセッションステートに存在しません。")

    response = None  # ここで response を初期化
    # プロンプト送信ボタン
    if st.button("プロンプト送信"):
        if not selected_checkbox:
            if not user_prompt.strip():
                st.error("プロンプトを入力してください。")
            else:
                if total_files != 0:
                    user_prompt = user_prompt + "\n\n  以下はアップロードされたファイルの内容：" + str(total_files)
                response = get_implementation_details(user_prompt, openai.api_type, OPENAI_CONFIG["model_name_llm"], conversation)
                st.success("プロンプトが正常に送信されました。")
        st.write(response)

#    if st.button("インデックス作成"):
#        #data = process_uploaded_files(uploaded_files, text_splitter)  # 新しい関数を呼び出す
#        # RAG処理を行う
#        # アップロードされたファイルからデータをロード
#        if uploaded_files:
#            data = process_uploaded_files(st.session_state['files_rag'], text_splitter)  # 新しい関数を呼び出す
#
#        if data:
#            # ベクトルストアを作成
#            vectors = FAISS.from_documents(data, embeddings)
#
#           # Conversational Retrieval Chainの作成
#            chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0, model_name=OPENAI_CONFIG["model_name"]), retriever=vectors.as_retriever())
#
#            # ユーザーからの質問を処理
#            result = chain({"question": user_prompt, "chat_history": st.session_state.get('rag_history', [])})
#            st.session_state['rag_history'] = st.session_state.get('rag_history', []) + [(user_prompt, result["answer"])]
#
#            # 分析したデータを指定されたフォルダに保存
#            output_path = OUTPUT_DIR_INDEX 
#            if not os.path.exists(output_path):
#                os.makedirs(output_path)
#            with open(os.path.join(output_path, "analysis_result.txt"), "w", encoding="utf-8") as f:
#                f.write(result["answer"])
#
#            st.write(result["answer"])

        # プロンプト送信時のトークンサイズとコストを表示
        token_size = st.session_state.prompt_tokens[-1][0]
        cost = st.session_state.prompt_tokens[-1][1]
        st.sidebar.write(f"今回送信時のトークン: {token_size} ")
        st.sidebar.write(f"コスト: ${cost:.6f}")

    # 質問と回答の履歴を表示
    if 'rag_conversation' not in st.session_state:
        st.session_state.rag_conversation = []

    if response:
        st.session_state.rag_conversation.append({"question": question, "answer": response})

    if st.session_state.rag_conversation:
        display_conversation(st.session_state.rag_conversation, 'rag')

# 質問履歴ページ
def display_question_history():
    st.title(PAGES["question_history"])

    st.write("### ホームページの質問履歴")
    # ホームページの質問履歴
    if st.session_state.home_conversation:
        display_conversation(st.session_state.home_conversation, 'home')

    st.write("*************************************************************************************************************************************")

    st.write("### 開発支援(JIRA連携)ページの質問履歴")
    # プロジェクト管理の質問履歴
    if st.session_state.project_conversation:
        display_conversation(st.session_state.project_conversation, 'project')

    st.write("*************************************************************************************************************************************")

    st.write("### RAG実験ページの質問履歴")
    # プロジェクト管理の質問履歴
    if st.session_state.rag_conversation:
        display_conversation(st.session_state.rag_conversation, 'rag')

    st.write("*************************************************************************************************************************************")

    # エクスポートボタン
    if st.button("CSV形式でエクスポート"):
        export_history_to_csv(st.session_state.home_conversation + st.session_state.project_conversation)
        st.success("CSVファイルがエクスポートされました。")

    #if st.button("PDF形式でエクスポート"):
    #    export_history_to_pdf(st.session_state.home_conversation + st.session_state.project_conversation)
    #    st.success("PDFファイルがエクスポートされました。")

    if st.button("HTML形式でエクスポート"):
        export_history_to_html(st.session_state.home_conversation + st.session_state.project_conversation)
        st.success("HTMLファイルがエクスポートされました。")

################################################################################################
# インデックス情報のリスト
if os.path.exists(INDEX_INFO_PATH):
    with open(INDEX_INFO_PATH, 'r', encoding='utf-8') as f:
        index_list = json.load(f)
else:
    index_list = []

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
    index = faiss.IndexFlatL2(OPENAI_CONFIG["dimension"])
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
    """指定されたファイルが既に処理済みかどうかを確認する"""
    return any(entry['file_path'] == file_path for entry in index_list)

# ベクトルデータ追加用関数の修正
def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    info_type = INFO_TYPES.get(ext, 'other')

    if is_file_processed(file_path):
        st.info(f"既に処理済みのファイルをスキップします: {file_path}")
        return

    content = ""
    if ext in ['.txt', '.py', '.md']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            st.warning(f"エンコーディングエラー: {file_path}")
            return
    elif ext == '.pdf':
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text
        except Exception as e:
            st.warning(f"PDFの読み込みに失敗しました: {file_path}")
            return
    elif ext == '.docx':
        try:
            doc = docx.Document(file_path)
            content = '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.warning(f"DOCXの読み込みに失敗しました: {file_path}")
            return
    elif ext in ['.xls', '.xlsx', '.xlsm']:
        try:
            content = read_excel_file(file_path)
        except Exception as e:
            st.warning(f"Excelファイルの読み込みに失敗しました: {file_path}")
            return
    else:
        st.info(f"サポートされていないファイルタイプをスキップします: {file_path}")
        return

    if not content.strip():
        st.info(f"空の内容をスキップします: {file_path}")
        return

    # コンテンツを分割してトークン数制限を超えないように処理
    chunks = text_splitter.split_text(content)

    total_tokens = sum(len(chunk) for chunk in chunks)

    # TOKEN_LIMITを超えた場合の処理
    if total_tokens > TOKEN_LIMIT:
        st.warning(f"ファイルがトークン数制限を超えています。分割して処理を行います。")

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

    st.success(f"ファイルを処理しました: {file_path}")
    
def process_zip(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                traverse_directory(tmpdirname)
    except Exception as e:
        st.warning(f"ZIPファイルの処理に失敗しました: {file_path}")

def traverse_directory(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext == '.zip':
                process_zip(file_path)
            else:
                process_file(file_path)

# ファイルパスに基づいてベクトルデータを取得する関数
def get_vector_from_faiss(file_path, vector_store, index_list):
    vectors = []
    #st.write(index_list)
#    try:
    for index in index_list:
        query_embedding = embeddings.embed_query(index["ids"])
        vdata = vector_store.similarity_search_by_vector(query_embedding, k=1)
        vectors.append(vdata[0].page_content)

    if not vectors:
        st.warning(f"{file_path} に対応するベクトルデータが見つかりませんでした。")
        return None
    return vectors
#    except Exception as e:
#        st.error(f"ベクトルデータの取得中にエラーが発生しました: {e}")
#        return None

# ベクトル取得とLLM問い合わせの処理を統合する関数
def vector_search_and_query_llm(prompt):
#    try:
    # クエリをベクトルに変換
    #st.write(prompt)
    query_embedding = embeddings.embed_query(prompt)

    # 類似度検索（ベクトル検索）
    # FAISSベクトルストアをロード
    #vector_store = FAISS.load(INDEX_FILE_PATH, embeddings)
    results = vector_store.similarity_search_by_vector(query_embedding, k=1)  # 最上位1つのみ取得
    #st.write(results)
    file_path = ""
    file_content = ""
    answer = ""
    if results:
        # 最上位のドキュメント内容を取得
        doc = results[0]
        #st.write(doc)
        file_path = doc.metadata["file_path"]

    #st.write(file_path)
    #st.write(len(index_list))
    index_list_by_file = []
    for index in index_list:
        if index["file_path"] == file_path:
            index_list_by_file.append(index)
    #st.write(len(index_list_by_file))
    # 指定されたファイルパスに基づいてベクトルを取得
    vectors = get_vector_from_faiss(file_path, vector_store, index_list_by_file)

    if vectors:
        # プロンプトとベクトルを基にLLMに問い合わせ
        llm_response = query_llm_with_vectors(vectors, prompt, OPENAI_CONFIG["model_name_llm"], conversation)

        if not llm_response:
            st.error("LLMからの回答が得られませんでした。")
    else:
        st.error("指定されたファイルパスに基づくベクトルが見つかりませんでした。")
    return file_path, llm_response
#    except Exception as e:
#        st.error(f"ベクトル検索またはLLM問い合わせの処理中にエラーが発生しました: {e}")
#        return None, None

# Streamlit検索ページ
def search_query(user_query):
    with st.spinner("検索中..."):
        # ベクトル検索とLLMとの連携結果を取得
        top_result, llm_response = vector_search_and_query_llm(user_query)

        if top_result:
            # 検索結果の表示
            st.subheader("最上位のドキュメント")
            st.write(top_result)

            # LLMからの応答を表示
            st.subheader("LLMからの回答")
            st.write(llm_response)
        else:
            st.warning(llm_response)
        return llm_response

# -------------------------
# Streamlit UI
# -------------------------

# サイドバーの設定
st.sidebar.title("メニュー")

# 現在のモードの状態をセッションステートで管理
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# トグルボタンを作成
dark_mode = st.sidebar.toggle("モード切替", value=st.session_state.dark_mode)

# トグルの状態に応じてモードを切り替える
if dark_mode != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_mode

# スタイルを適用
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #2E2E2E;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {
            background-color: white;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)

#def count_tokens(text):
#    tokens = tokenizer.encode(text)
#    return len(tokens)

# PDFファイルのテキストを読み込んでトークン数を確認
#def count_tokens_in_file(file_path):
#    reader = PdfReader(file_path)
#    content = ''.join([page.extract_text() or '' for page in reader.pages])
#    return count_tokens(content)

################################################################################################
def display_file_process():
    st.header("ファイルのアップロードと処理")

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("ファイルアップロード")
        uploaded_files = st.file_uploader(
            "ここにファイルをアップロードしてください（複数可）",
            type=['txt', 'py', 'pdf', 'docx', 'zip', 'md', 'xls', 'xlsx', 'xlsm'],
            accept_multiple_files=True
        )
        process_button = st.button("ファイルを処理する")

    with col2:
        st.subheader("処理の進捗")
        progress_bar = st.progress(0)
        status_text = st.empty()

    st.markdown("---")

    # フォルダーの処理
    folder_path = st.text_input(
        "処理するフォルダーのパスを入力してください:",
        value=INPUT_DIR_INDEX,
        help="指定されたフォルダー内のファイルも再帰的に処理されます。"
    )
    process_folder_button = st.button("フォルダーを処理する")

    if process_button and uploaded_files:
        with st.spinner("ファイルを処理中..."):
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
                    st.write(f"**{uploaded_file.name}** の内容:\n{content}")
                else:
                    process_file(tmp_file_path)

                # プログレスバーの更新
                progress = (idx + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"処理中: {uploaded_file.name}")

            st.success("アップロードされた全てのファイルの処理が完了しました。")

    elif process_button:
        st.warning("処理するファイルをアップロードしてください。")

    if process_folder_button and folder_path:
        if os.path.isdir(folder_path):
            with st.spinner("フォルダーを処理中..."):
                traverse_directory(folder_path)
                st.success(f"フォルダー '{folder_path}' の処理が完了しました。")
                save_index(vector_store, index_list)  # フォルダー処理が完了したらインデックスを保存する
        else:
            st.error(f"指定されたフォルダーが存在しません: {folder_path}")
    elif process_folder_button:
        st.warning("処理するフォルダーのパスを入力してください。")

# 既存の display_session_data() 関数を改良
def display_session_data():
    st.title("SessionData")
    
    if st.button("セッションデータをダイレクトインポート  \n\n  "):
        import_session_data_direct()
        st.success("セッションデータがインポートされました。")

    #if st.button("セッションデータをエクスポート"):
    #    export_session_data()
    #    st.success("セッションデータがエクスポートされました。")

    #st.write("  \n\n  ")
    #import_session_data()  # 改良されたインポート機能を呼び出す
    st.write("  \n\n  ")
    #st.write(st.session_state)
    # 折り畳み表示
    with st.expander('Session Data  \n\n  '):
        st.json(st.session_state)

# これまでに送ったプロンプト送信時のトークンサイズと総計コストを表示
def display_total_tokens_and_cost():
    total_tokens = sum(tokens[0] for tokens in st.session_state.prompt_tokens)
    st.sidebar.write(f"累計送受信トークン: {total_tokens:,} ")
    st.sidebar.write(f"総計コスト: ${st.session_state.total_cost:.6f}")

# セッションサイズ表示
def display_session_size():
    session_size = sum(len(str(value)) for value in st.session_state.values())
    st.sidebar.write(f"現在のセッションサイズ: {session_size:,} バイト")

# セッションをクリアする関数
def clear_session():
    st.session_state.clear()  # セッション状態をクリアする

if __name__ == "__main__":
    main()

# -------------------------
# 終了
# -------------------------
st.sidebar.markdown("---")
st.sidebar.write("© 2024 RAGアプリケーション")