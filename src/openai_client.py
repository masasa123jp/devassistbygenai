# -*- coding: utf-8 -*-
import openai
import streamlit as st
from functools import wraps
import os
from openai import AzureOpenAI
import httpx
from src.error_handler import handle_api_errors
from src.config import OPENAI_CONFIG, set_token_cost

#OPENAI_CONFIG = {
#    'api_base': 'https://api.openai.com/v1',
#    'api_key': os.environ.get('OPENAI_API_KEY'),
#    'model_name': 'gpt-4o-mini',
#}

# トークンコストの定義  # 1トークンあたりのコスト
#TOKEN_COSTS = {
#    'gpt-o1-mini' :
#        {'input_gpt-o1-mini': 0.000003,'output_gpt-o1-mini': 0.000012},
#    'gpt-4o-mini' :
#        {'input_gpt-4o-mini': 0.00000015,'output_gpt-4o-mini': 0.0000006},
#    'text-embedding-3-small' :
#        {'input_text-embedding-3-small': 0.00000002,'output_text-embedding-3-small': 0.00000002}
#}

# トークンコストの設定
#def set_token_cost(model_name):
#    if model_name in TOKEN_COSTS:
#        return TOKEN_COSTS[model_name][f'input_{model_name}'], TOKEN_COSTS[model_name][f'output_{model_name}']
#    return 0.0, 0.0

TOKEN_COST_INPUT, TOKEN_COST_OUTPUT = set_token_cost(OPENAI_CONFIG['model_name_llm'])

# 初期化
@handle_api_errors
def configure_openai():
    openai.api_type = 'openai'
    openai.api_key = OPENAI_CONFIG['api_key']
    openai.api_base = OPENAI_CONFIG['api_base']

# LLMとの対話
# OpenAI APIの設定
def get_implementation_details(prompt, api_type, model_name, conversation):
    client = None
    if api_type == 'openai':
        client = openai
    elif api_type == 'azure':
        client = AzureOpenAI(
            api_version=openai.api_version,
            http_client=httpx.Client(verify=False)
        )

    # APIリクエストのためのメッセージ構造を準備
    messages = [{'role': 'user', 'content': prompt}]

    # conversationオブジェクトが存在し、その中のmemory.messagesが存在する場合に履歴を追加
#    if conversation and conversation.memory and conversation.memory.messages:
#        for message in conversation.memory.messages: .chat_memory.messages
    if conversation and conversation.memory and conversation.memory.chat_memory.messages:
        messages = conversation.memory.chat_memory.messages  # ここでget_messages()メソッドでメッセージを取得する
        if messages:  # メッセージが存在するか確認
            for message in messages:
                messages.insert(0, {'role': 'assistant' if message['role'] == 'user' else 'user', 'content': message['content']})

    response = openai.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7
    )

    cost_input = response.usage.prompt_tokens * TOKEN_COST_INPUT
    cost_output = (response.usage.total_tokens - response.usage.prompt_tokens) * TOKEN_COST_OUTPUT
    token_count = response.usage.total_tokens
    cost = cost_input + cost_output
    st.session_state.prompt_tokens.append((token_count, cost))
    st.session_state.total_cost += cost
    return response.choices[0].message.content