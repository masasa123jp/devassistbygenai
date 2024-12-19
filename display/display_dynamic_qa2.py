# -*- coding: utf-8 -*-
# src/display/display_dynamic_qa2.py
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from src.config import OPENAI_CONFIG
import openai
from src.openai_client import get_implementation_details

def display_dynamic_qa2(conversation: ConversationalRetrievalChain):
    st.header("ダイナミック質問生成")

    # ユーザーからトピックを取得
    topic = st.text_area('関心のあるトピックを入力してください:')

    llm = ChatOpenAI(temperature=0, model_name=OPENAI_CONFIG['model_name_llm'])

    if st.button('質問を生成'):
        if topic:
            input_message = f'このトピックに関する情報を教えてください: {topic}'
            response = llm([{"role": "user", "content": input_message}])

            # 生成された質問を表示
            generated_questions = [
                f'{topic}についてどう思いますか？',
                f'{topic}に関する具体例を挙げてください。',
                f'{topic}についてのあなたの意見は何ですか？'
            ]

            st.subheader('生成された質問')
            for question in generated_questions:
                st.write(f'- {question}')

            user_response = st.text_area('質問に対するあなたの答えを入力してください:')

            if st.button('送信'):
                if user_response:
                    follow_up_response = conversation({'question': f'{user_response}についてもっと教えてください。', 'chat_history': []})
                    st.subheader('AIからの応答')
                    st.write(follow_up_response['answer'])
                else:
                    st.error('応答を入力してください。')
        else:
            st.error('トピックを入力してください。')

def generate_response_with_history(user_message, history):
    prompt = "以下の会話履歴に基づいて回答してください:\n"
    for entry in history:
        prompt += f"ユーザー: {entry['user']}\nAI: {entry['ai']}\n"
    prompt += f"ユーザー: {user_message}\nAI:"

    response = openai.chat.completions.create(
        model=OPENAI_CONFIG['model_name_llm'],
        messages=[{"role": "system", "content": "あなたは親切なAIアシスタントです。"},
                  {"role": "user", "content": prompt}]
    )

    return response['choices'][0].message.content

def vector_search_and_query_llm_with_rag(prompt, embeddings, vector_store):
    query_embedding = embeddings.embed_query(prompt)
    results = vector_store.similarity_search_by_vector(query_embedding, k=5)  # 上位5つを取得

    if results:
        vectors = [result.page_content for result in results]
        llm_response = query_llm_with_vectors(vectors, prompt, OPENAI_CONFIG['model_name_llm'])
        return llm_response
    return "関連する情報が見つかりませんでした。"

# 取得したベクトルを使用してLLMに問い合わせる関数
def query_llm_with_vectors(vectors, prompt, llm_model, conversation):
#    try:
    # ベクトルデータを準備し、LLM用のプロンプトを作成
    vector_content = '\n'.join([str(vector) for vector in vectors])
    complete_prompt = f"以下のベクトルデータに基づいて、次のプロンプトに回答してください:\nベクトルデータ:\n{vector_content}\n\nプロンプト: {prompt}"
    #st.write(complete_prompt)

    # OpenAI LLMに問い合わせ
#    response = openai.ChatCompletion.create(
#        model=llm_model,
#        messages=[{"role": "system", "content": "あなたは高度な情報処理システムです。"},
#                    {"role": "user", "content": complete_prompt}],
#        max_tokens=1000
#    )
    
    # get_implementation_detailsを使ってLLMに問い合わせ
    response = get_implementation_details(complete_prompt, openai.api_type, llm_model, conversation)
#    response = openai.chat.completions.create(
#        model=llm_model,
#        messages=[{"role": "system", "content": "あなたは高度な情報処理システムです。"},
#                    {"role": "user", "content": complete_prompt}],
#        temperature=0.7
#    )
    #return response['choices'][0]['message']['content']
    return response.choices[0].message.content
#    except Exception as e:
#        st.error(f"LLMへの問い合わせ中にエラーが発生しました: {e}")
#        return None