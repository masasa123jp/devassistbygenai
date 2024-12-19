# -*- coding: utf-8 -*-
import streamlit as st
import openai

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationChain

from src.openai_client import get_implementation_details
from src.conversation import display_conversation
from src.config import OPENAI_CONFIG, PAGES
from src.file_processing import read_uploaded_files,read_files_from_directory_as_str
from langchain.memory import ConversationBufferMemory

ANALYSYS_FOLDER_PATH =r'D:/pg/py/3.2.15/DevAssistByGenAI'
EXTENSIONS_TO_SEARCH = ['.md', '.py','php', '.bat']  # 検索したい拡張子リストを指定

# ホームページ
def display_home(data:str, conversation:ConversationChain):
    # ホームページ
    st.title(PAGES['home'])

    # チェックボックスの定義
    prompt_program = 'このアプリの使い方を教えてください。'
    selected_checkbox = st.checkbox(label=prompt_program, value=st.session_state.home_checkbox_state)
    prompt_program2 = 'このアプリについて'
    selected_checkbox2 = st.checkbox(label=prompt_program2 + '(こちらは、上のチェックボックスは外して利用する事)：', value=st.session_state.home_checkbox_state2)
    prompt_program3 = '指定フォルダ配下のアプリ使い方を教えてください。（PJ）: '
    selected_checkbox3 = st.checkbox(label=prompt_program3 + '(こちらは、上のチェックボックスは外して利用する事)：', value=st.session_state.home_checkbox_state3)
    textarea_folder = st.text_input('フォルダを指定してください。:', value=ANALYSYS_FOLDER_PATH)

    data_py_pj = ''
    # Pythonプロジェクトデータ取得ボタン
    if st.button('データ取得'):
        st.write(textarea_folder)
        data_py_pj = read_files_from_directory_as_str(textarea_folder, EXTENSIONS_TO_SEARCH)
        st.write(len(data_py_pj))
        #st.write(data_py_pj[:500])

    user_prompt = ''
    question = ''
    # プロンプトを入力/作成
    if not selected_checkbox:
        if selected_checkbox2:
            textarea = st.text_area('プロンプトを入力してください HOME:', height=200) 
            user_prompt = textarea + 'ただし、認証情報の文字列は伏せてください。:  \r\n  ' + data
            question = textarea
            st.session_state.home_checkbox_state = False
            st.session_state.home_checkbox_state3 = False
        elif selected_checkbox3 == True :
            textarea = st.text_area('プロンプトを入力してくださいHOME:', height=200, value='指定フォルダ配下のアプリの使い方やモジュールの構成を教えてください。また、このアプリの特徴や特長を教えてください。') 
            user_prompt = textarea + '  \r\n  ' + data_py_pj
            question = prompt_program3 + textarea_folder
            st.session_state.checkbox_state = False
            st.session_state.checkbox_state2 = False
        else:
            user_prompt = st.text_area('プロンプトを入力してください HOME:', height=200)
            question = user_prompt
    else:
        user_prompt = prompt_program + 'ただし、認証情報の文字列は伏せてください。また実装の具体的、詳細な解説はしないでください。:  \r\n  ' +  data
        question = prompt_program
        st.session_state.home_checkbox_state2 = False
        st.session_state.home_checkbox_state3 = False

    uploaded_files = st.file_uploader('ファイルをアップロードしてください (最大4ファイル) HOME', 
                                      type=['txt', 'pdf', 'docx', 'xls', 'xlsx', 'xlsm', 'py', 'zip', 'md'], accept_multiple_files=True)

    # アップロードされたファイルの内容を読み込む
    file_content = ''
    if uploaded_files:
        file_content = read_uploaded_files(uploaded_files)
        if file_content:
            st.success('ファイルが正常にアップロードされました。')
        else:
            st.error('ファイルのアップロードに失敗しました。')
    #st.write(user_prompt)

    response = None
    # プロンプト送信ボタン
    if st.button('プロンプト送信'):
        data_py_pj = ''
        #st.write('button')
        if selected_checkbox3 and textarea_folder != '':
            data_py_pj = read_files_from_directory_as_str(textarea_folder, EXTENSIONS_TO_SEARCH)
            st.write(len(data_py_pj))

        if not selected_checkbox:
            #st.write('test')
            if not user_prompt.strip():
                st.error('プロンプトを入力してください。')
            else:
                filedata = ''
                if file_content:
                    for key in file_content.keys():
                        filedata += '  \n  '+ file_content[key]
                user_prompt = user_prompt + data_py_pj + '\n\n  以下はアップロードされたファイルの内容：' + filedata
                #st.write('test')
                response = get_implementation_details(user_prompt, openai.api_type, OPENAI_CONFIG['model_name_llm'], conversation)
                st.success('プロンプトが正常に送信されました。')
        else:
            response = get_implementation_details(user_prompt, openai.api_type, OPENAI_CONFIG['model_name_llm'], conversation)
            st.success('プロンプトが正常に送信されました。')

        st.write(response)
        # ここでメモリに会話を追加
        conversation.memory.save_context({'input': user_prompt}, {'output': response})

        # プロンプト送信時のトークンサイズとコストを表示
        if st.session_state.prompt_tokens:  # リストが空でないことを確認
            token_size = st.session_state.prompt_tokens[-1][0]
            cost = st.session_state.prompt_tokens[-1][1]
            st.sidebar.write(f'今回送信時のトークン: {token_size} ')
            st.sidebar.write(f'コスト: ${cost:.6f}')
        else:
            st.sidebar.write('今回送信時のトークン情報がありません。')

    # UI部分での利用
    #if st.button('コードを生成'):
    #    if user_prompt.strip():
    #        code_response = generate_code_from_natural_language(user_prompt)
    #        st.code(code_response, language='python')
    #    else:
    #        st.error('要件を入力してください。')

    # UI部分での利用
    #if st.button('APIドキュメントを生成'):
    #    api_details = {
    #        'endpoint': '/api/v1/example',
    #        'method': 'POST',
    #        'description': 'このAPIは例を処理します。',
    #        'parameters': 'param1: String, param2: Integer',
    #        'example': "{'param1': 'value1', 'param2': 123}"
    #    }
    #    doc_response = generate_api_documentation(api_details)
    #    st.markdown(doc_response)

    # UI部分での利用
    #if st.button('コードレビューを依頼'):
    #    data_py_pj = read_files_from_directory_as_str(textarea_folder, EXTENSIONS_TO_SEARCH)
    #    if data_py_pj.strip():
    #        review_response = review_code(data_py_pj)
    #        st.write(review_response)
    #    else:
    #        st.error('レビューしたいコードを入力してください。')

    # 質問と回答の履歴を表示
    if 'home_conversation' not in st.session_state:
        st.session_state.home_conversation = []

    if response:
        st.session_state.home_conversation.append({'question': question, 'answer': response})

    if st.session_state.home_conversation:
        # 質問履歴のフィルタリング機能の追加
        filter_topic = st.text_input('フィルタリングしたいトピックを入力してくださいHOME:')
        filtered_history = [entry for entry in st.session_state.home_conversation if filter_topic.lower() in entry['question'].lower()]

        if filtered_history:
            st.write('フィルタリングされた質問履歴:')
            display_conversation(filtered_history, 'home')
        else:
            st.write('フィルタリングされた結果はありません。')
         #display_conversation(st.session_state.home_conversation, 'home')

# 新しい関数を追加
def generate_code_from_natural_language(prompt):
    completion = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'user', 'content': f'次の要件に基づいてPythonのコードを生成してください: {prompt}'}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content

def generate_api_documentation(api_details):
    documentation = f'''
    # API Documentation

    ## Endpoint: {api_details['endpoint']}

    ### Method: {api_details['method']}

    ### Description:
    {api_details['description']}

    ### Parameters:
    - {api_details['parameters']}

    ### Example:
    ```
    {api_details['example']}
    ```
    '''
    return documentation


def review_code(code):
    completion = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'user', 'content': f'次のコードをレビューして、改善点を教えてください:\n{code}'}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content

