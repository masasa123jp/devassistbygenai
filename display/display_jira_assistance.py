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

from src.error_handler import handle_api_errors
from src.jira_client import create_jira_client, get_all_projects, get_user_backlog
from src.openai_client import configure_openai, get_implementation_details
from src.config import PAGES, JIRA_CONFIG, OPENAI_CONFIG, PROMPT_PRE_OUTSYSTEMS
from src.file_processing import read_uploaded_files
from src.conversation import display_conversation

# 開発支援(JIRA連携)画面
def display_jira_assistance(conversation:ConversationChain):
    st.title(PAGES['assist_page_ref_jira'])

    # ユーザーアカウントIDの入力
    st.session_state.user_account_id = st.text_input('JIRAユーザーアカウントIDを入力:', value=JIRA_CONFIG['user_name'])

    # JIRAクライアントの作成
    jira_client = create_jira_client()
    
    projects =  []
    #st.write(st.session_state.projects)
     # プロジェクトの選択
    if len(st.session_state.projects) != 0:
        project_list = []
        for project in st.session_state.projects:
            if isinstance(project, str):
                try:
                    key = project.split(',')[0].split('key=')[1].strip()[1:-1]
                    value = project.split(',')[1].split('name_project=')[1].strip()[1:-2]
                    append_project = Project(key, value)
                    project_list.append(append_project)
                except IndexError as e:
                    st.error(f'プロジェクトの解析中にエラーが発生しました: {str(e)}')

        if len(project_list) != 0:
            projects = project_list
    else :
        project_list = get_all_projects(jira_client)
        projects = [project for project in project_list if project.key != 'ORCASUP']
        st.session_state.projects = projects
    
    #st.write(st.session_state.projects)

    # プロジェクトの選択
    project_options = [f'{project.key}: {project.name_project}' for project in st.session_state.projects]
    if len(st.session_state.projects) != 0:
        if 'selected_project' not in st.session_state:
            st.session_state.selected_project = project_options[0]
        
        selected_project = st.radio('プロジェクトを選択:', [project for project in project_options])

        if st.session_state.selected_project:
            if selected_project.split(':')[0].strip() != st.session_state.selected_project.split(':')[0].strip():
                st.session_state.project_prompt = PROMPT_PRE_OUTSYSTEMS
                st.session_state.backlog_issues = []
        
        st.session_state.selected_project = selected_project
        selected_project_key = st.session_state.selected_project.split(':')[0]
        st.session_state.selected_project_obj = next((proj for proj in st.session_state.projects if proj.key == selected_project_key), None)

    backlog_issues = []
    # バックログ取得ボタン
    if st.button('バックログ取得'):
        # バックログの取得処理
        backlog_issues = get_user_backlog(jira_client, st.session_state.selected_project_obj.key, st.session_state.user_account_id)
        st.session_state.backlog_issues = sorted(backlog_issues, key=lambda x: x.key)
        if len(backlog_issues) == 0:
            st.write('対象バックログがありません')

    if len(backlog_issues) != 0:
        issue_list = []
        for issue in st.session_state.backlog_issues:
            #st.write(issue)
            if isinstance(issue, str):
                try:
                    key = issue.split('key=')[1].split("',")[0].strip()[1:]
                    name_issue = issue.split('name_issue=')[1].split("',")[0].strip()[1:]
                    task_description = issue.split('task_description=')[1].split("',")[0].strip()[1:]
                    status = issue.split('status=')[1].split("',")[0].strip()[1:]
                    assignee = issue.split('assignee=')[1].split("',")[0].strip()[1:]
                    append_issue = Issue(key, name_issue, task_description, status, assignee)
                    #st.write(append_issue)
                    issue_list.append(append_issue)
                except IndexError as e:
                    st.error(f'プロジェクトの解析中にエラーが発生しました: {str(e)}')

        if len(issue_list) != 0:
            st.session_state.backlog_issues = issue_list

        selected_issue_key = st.radio('バックログを選択:', [f'{issue.key} : {issue.name_issue}' for issue in st.session_state.backlog_issues])
        selected_issue_key = selected_issue_key.split(':')[0].strip()
        st.session_state.selected_issue = next((issue for issue in st.session_state.backlog_issues if issue.key.strip() == selected_issue_key.strip()), None)
        project_prompt = PROMPT_PRE_OUTSYSTEMS + st.session_state.selected_issue.task_description
    elif st.session_state.selected_issue:
        issue_list = []
        for issue in st.session_state.backlog_issues:
            #st.write(issue)
            if isinstance(issue, str):
                try:
                    key = issue.split('key=')[1].split("',")[0].strip()[1:]
                    name_issue = issue.split('name_issue=')[1].split("',")[0].strip()[1:]
                    task_description = issue.split('task_description=')[1].split("',")[0].strip()[1:]
                    status = issue.split('status=')[1].split("',")[0].strip()[1:]
                    assignee = issue.split('assignee=')[1].split("',")[0].strip()[1:]
                    append_issue = Issue(key, name_issue, task_description, status, assignee)
                    #st.write(append_issue)
                    issue_list.append(append_issue)
                except IndexError as e:
                    st.error(f'プロジェクトの解析中にエラーが発生しました: {str(e)}')

        if len(issue_list) != 0:
            st.session_state.backlog_issues = issue_list

        #st.write(st.session_state.backlog_issues)

        selected_issue_key = st.radio('バックログを選択:', [f'{issue.key} : {issue.name_issue}' for issue in st.session_state.backlog_issues])
        selected_issue_key = str(selected_issue_key).split(':')[0].strip()
        st.session_state.selected_issue = next((issue for issue in st.session_state.backlog_issues if issue.key.strip() == selected_issue_key.strip()), None)
        #st.write(selected_issue_key)
        #st.write(st.session_state.selected_issue)
        if st.session_state.selected_issue != None:
            project_prompt = PROMPT_PRE_OUTSYSTEMS + st.session_state.selected_issue.task_description
        else:
            project_prompt = PROMPT_PRE_OUTSYSTEMS
    else:
        project_prompt = PROMPT_PRE_OUTSYSTEMS

    # テキストエリアとファイルアップロード
    st.session_state.text_area_value = st.text_area('詳細情報:OutSytemsで実装する方法を調査する内容を入力:', value=project_prompt, height=150)
    
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    
    uploaded_files = st.file_uploader('ファイルをアップロードしてください (最大4ファイル) JIRA', type=['txt', 'pdf', 'docx', 'py', 'zip', 'md'], accept_multiple_files=True)

    # アップロードされたファイルの内容を読み込む
    file_content = ''
    if uploaded_files:
        file_content = read_uploaded_files(uploaded_files)

    # プロンプト送信ボタン
    if st.button('プロンプト送信', key='project_prompt_button'):
        # ファイル内容をプロンプトに追加
        st.session_state.text_area_value += '\n\n' + file_content

        response_project = get_implementation_details(project_prompt, openai.api_type, OPENAI_CONFIG['model_name_llm'], conversation)

        st.session_state.project_conversation.append({
            'question': st.session_state.text_area_value,
            'answer': response_project
        })
        st.write(response_project)

        # プロンプト送信時のトークンサイズとコストを表示
        token_size = st.session_state.prompt_tokens[-1][0]
        cost = st.session_state.prompt_tokens[-1][1]
        st.sidebar.write(f'今回送信時のトークン: {token_size} ')
        st.sidebar.write(f'コスト: ${cost:.6f}')

    # 質問と回答の履歴を表示
    if 'project_conversation' not in st.session_state:
        st.session_state.project_conversation = []
    if st.session_state.project_conversation:
        # 質問履歴のフィルタリング機能の追加
        filter_topic = st.text_input('フィルタリングしたいトピックを入力してくださいJIRA:')
        filtered_history = [entry for entry in st.session_state.project_conversation if filter_topic.lower() in entry['question'].lower()]

        if filtered_history:
            st.write('フィルタリングされた質問履歴:')
            display_conversation(filtered_history, 'project')
        else:
            st.write('フィルタリングされた結果はありません。')
        #display_conversation(st.session_state.project_conversation, 'project')
