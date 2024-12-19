# -*- coding: utf-8 -*-
import streamlit as st

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

from src.config import PAGES,OPENAI_CONFIG
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


# 新しいダイナミックQ&A機能を追加するための関数
def display_dynamic_qa1(conversation: ConversationalRetrievalChain):
    st.header(PAGES['dynamic_qa'])

    # ユーザーからトピックを取得
    topic = st.text_area('関心のあるトピックを入力してください:')

    # ChatOpenAIを使用して会話を行う
    llm = ChatOpenAI(temperature=0, model_name=OPENAI_CONFIG['model_name_llm'])


    if st.button('質問を生成'):
        if topic:
            # トピックに基づく情報を取得
            # 会話の履歴を構築するためのリスト
            chat_history = []

            # 質問を追加する
            input = [HumanMessage(content=f'このトピックに関する情報を教えてください: {topic}')]

            # 会話を処理する
            response = llm(input)
            #st.write(response)

            # 応答を取得する
            answer = response.content

            # 取得した情報に基づいて質問を生成
            generated_questions = [
                f'{topic}についてどう思いますか？',
                f'{topic}に関する具体例を挙げてください。',
                f'{topic}についてのあなたの意見は何ですか？'
            ]

            # 生成された質問を表示
            st.subheader('生成された質問')
            for question in generated_questions:
                st.write(f'- {question}')

            # ユーザーが質問に答えるためのテキストエリア
            user_response = st.text_area('質問に対するあなたの答えを入力してください:')

            if st.button('送信'):
                if user_response:
                    # ユーザーの応答を基にAIがさらなる情報を提供
                    follow_up_response = conversation({'question': f'{user_response}についてもっと教えてください。', 'chat_history': []})
                    st.subheader('AIからの応答')
                    st.write(follow_up_response['answer'])
                else:
                    st.error('応答を入力してください。')
        else:
            st.error('トピックを入力してください。')
