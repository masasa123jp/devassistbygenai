# -*- coding: utf-8 -*-
import streamlit as st
# 質問と回答を表示する関数
def display_conversation(conversation, conversation_type):
    """
    質問と回答を表示する関数

    :param conversation: 質問と回答のリスト
    :param conversation_type: 質問履歴の種類
    """
    for idx, qa in enumerate(conversation):
        with st.expander(f"質問 {idx + 1}: {qa['question']}"):
            st.write(f"**質問:** {qa['question']}")
            st.write(f"**回答:** {qa['answer']}")
            if st.button("削除", key=f"delete_{conversation_type}_{idx}"):
                if conversation_type == 'home':
                    st.session_state.home_conversation.pop(idx)
                elif conversation_type == 'project':
                    st.session_state.project_conversation.pop(idx)
                elif conversation_type == 'rag':
                    st.session_state.rag_conversation.pop(idx)
                st.success("質問が削除されました。")

# 指定された会話の削除を処理する関数
def handle_conversation_deletion(conversation_type, index):
    """
    指定された会話の削除を処理する関数

    :param conversation_type: 削除する会話の種類
    :param index: 削除する会話のインデックス
    """
    if conversation_type == 'home':
        st.session_state.home_conversation.pop(index)
    elif conversation_type == 'project':
        st.session_state.project_conversation.pop(index)
    elif conversation_type == 'rag':
        st.session_state.rag_conversation.pop(index)
    st.success("質問が削除されました。")
