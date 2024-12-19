# error_handler.py
import streamlit as st
from functools import wraps

def handle_api_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f'エラーが発生しました。関数: {func.__name__} | エラー内容: {str(e)}')
            return None
    return wrapper