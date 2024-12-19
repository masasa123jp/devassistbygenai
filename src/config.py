# -*- coding: utf-8 -*-
# config.py
import os
import PyPDF2
import docx

# JIRAの設定
JIRA_CONFIG = {
    'server': 'https://XXXXXXXXXXX.atlassian.net/',
    'user_name': 'XXXXXXXXXXX@gmail.com',
    'user_name_for_access': 'XXXXXXXXXXX@XXXXXXXXXXXXXXX',
    'api_token': 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
}

# OpenAI APIの設定
OPENAI_CONFIG = {
    'api_base': 'https://api.openai.com/v1',
    'api_key': os.environ.get('OPENAI_API_KEY'),
    'model_name_llm': 'gpt-4o-mini',
    'model_name_embeddings': 'text-embedding-3-small',
    'dimension': 1536,
}

# データ保存ディレクトリの設定
INPUT_DIR = os.environ.get('FILE_DIR', r'X:/pg/data/tool_data/')
OUTPUT_DIR = os.environ.get('FILE_DIR', r'X:/pg/data/tool_data/')

TARGET_EXTENSIONS = ('.txt', '.pdf', '.docx', '.py', '.zip', '.md', '.xls', '.xlsx', '.xlsm')

PROMPT_PRE_OUTSYSTEMS = '以下をOutSystemsによる画面やファイルアップロードフォーム（必要に応じた）を持つWebアプリケーションとしてローコードに実装する手順や方法を詳細に教えてください。:  \n   '

# データ保存ディレクトリの設定
INPUT_DIR_INDEX = r'X:/pg/data/tool_data/RAG'
OUTPUT_DIR_INDEX = r'X:/pg/data/tool_data/RAG_ANALYSIS'

# OpenAIの埋め込みモデルにおけるトークン数の制限
TOKEN_LIMIT = 1000000

if not os.path.exists(OUTPUT_DIR_INDEX):
    os.makedirs(OUTPUT_DIR_INDEX)

# インデックスファイルのパス
INDEX_FILE_PATH = os.path.join(OUTPUT_DIR_INDEX, 'faiss_index')
# インデックス情報の保存ファイル
INDEX_INFO_PATH = os.path.join(OUTPUT_DIR_INDEX, 'index.json')

# トークンコストの定義  # 1トークンあたりのコスト
TOKEN_COSTS = {
    'gpt-o1-mini' :
        {'input_gpt-o1-mini': 0.000003,'output_gpt-o1-mini': 0.000012},
    'gpt-4o-mini' :
        {'input_gpt-4o-mini': 0.00000015,'output_gpt-4o-mini': 0.0000006},
    'text-embedding-3-small' :
        {'input_text-embedding-3-small': 0.00000002,'output_text-embedding-3-small': 0.00000002}
}

# トークンコストの設定
def set_token_cost(model_name):
    if model_name in TOKEN_COSTS:
        return TOKEN_COSTS[model_name][f'input_{model_name}'], TOKEN_COSTS[model_name][f'output_{model_name}']
    return 0.0, 0.0

TOKEN_COST_INPUT_LLM, TOKEN_COST_OUTPUT_LLM = set_token_cost(OPENAI_CONFIG['model_name_llm'])
TOKEN_COST_INPUT_EMBEDDED_LLM, TOKEN_COST_OUTPUT_EMBEDDED_LLM = set_token_cost(OPENAI_CONFIG['model_name_llm'])

# ファイルごとのMIMEタイプ
FILE_TYPE ={
    'txt' : 'text/plain',
    'pdf' : 'application/pdf',
    'docx' : 'application/vnd.openxmlformats-officedocument.spreadsheetml.document',
    'xls' : 'application/vnd.ms-excel',
    'xlsx' : 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'xlsm' : 'application/vnd.ms-excel',
    'py' : 'text/x-python',
    'zip' : 'application/x-zip-compressed',
    'md' : 'application/octed-stream'
}

# 対応する拡張子ごとの情報タイプ
INFO_TYPES = {
    '.txt': 'text',
    '.py': 'code',
    '.pdf': 'document',
    '.docx': 'document',
    '.md': 'markdown',
    '.xls': 'document',
    '.xlsx': 'document',
    '.xlsm': 'document',
}

# ページ定義の追加
PAGES = {
    'home': 'ホーム',
    'assist_page_ref_jira': '開発支援(JIRA連携)',
    #'RAG': 'RAG実験',
    'dynamic_qa': 'ダイナミックQ&A',
    'question_history': '質問履歴',
    #'file_process' : 'ファイル処理',
    'search_and_file_processing' : '情報検索 and ファイル処理',
    'preference' : '設定',
    'SessionData': 'SessionData'
}
