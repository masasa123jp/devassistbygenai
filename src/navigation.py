# -*- coding: utf-8 -*-
# navigation.py
import streamlit as st
from src.config import PAGES

# ナビゲーションメニュー
def display_navigation_menu():
    st.sidebar.title('開発支援 by Gen-AI \n\n ナビゲーション')
    # 定数の追加
    return st.sidebar.radio('選択', list(PAGES.values()))