# -*- coding: utf-8 -*-
import streamlit as st
import os
import json

from src.error_handler import handle_api_errors

OUTPUT_DIR = os.environ.get('FILE_DIR', r'D:/pg/data/tool_data/')
# セッションデータのインポート関数
@handle_api_errors
def import_session_data_direct(filename=OUTPUT_DIR + 'session_data.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        st.session_state.update(json.load(f))

# 既存の display_session_data() 関数を改良
def display_session_data():
    st.title('SessionData')
    
    if st.button('セッションデータをダイレクトインポート  \n\n  '):
        import_session_data_direct()
        st.success('セッションデータがインポートされました。')

    #if st.button('セッションデータをエクスポート'):
    #    export_session_data()
    #    st.success('セッションデータがエクスポートされました。')

    #st.write('  \n\n  ')
    #import_session_data()  # 改良されたインポート機能を呼び出す
    st.write('  \n\n  ')
    #st.write(st.session_state)
    # 折り畳み表示
    with st.expander('Session Data  \n\n  '):
        st.json(st.session_state)
