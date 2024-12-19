# -*- coding: utf-8 -*-
import streamlit as st
import pandas
import io
import json
from src.config import OUTPUT_DIR
from src.error_handler import handle_api_errors

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


# 質問履歴をCSV形式でエクスポートする関数
@handle_api_errors
def export_history_to_csv(history, filename=OUTPUT_DIR + 'question_history.csv'):
    df = pandas.DataFrame(history)
    df.to_csv(filename, index=False, encoding='utf-8-sig')


# フォントの設定
styles = getSampleStyleSheet()
styles['Normal'].fontSize = 10

# 質問履歴をPDF形式でエクスポートする関数
@handle_api_errors
def export_history_to_pdf(history, filename=OUTPUT_DIR + 'question_history.pdf'):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = [Paragraph('質問履歴', styles['Title']), Spacer(1, 12)]

    for item in history:
        question = Paragraph(f'質問: {item['question']}', styles['Normal'])
        answer = Paragraph(f'回答: {item['answer']}', styles['Normal'])
        story.extend([question, answer, Spacer(1, 12)])

    doc.build(story)
    buffer.seek(0)

    with open(filename, 'wb') as f:
        f.write(buffer.getvalue())

# 質問履歴をHTML形式でエクスポートする関数
@handle_api_errors
def export_history_to_html(history, filename= OUTPUT_DIR +'question_history.html'):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('<html><head><title>質問履歴</title></head><body>')
        f.write('<h1>質問履歴</h1>')
        for item in history:
            f.write(f'<h2>質問: {item['question']}</h2>')
            f.write(f'<p>回答: {item['answer']}</p>')
            f.write('<hr>')
        f.write('</body></html>')

# セッションデータのエクスポート関数
@handle_api_errors
def export_session_data(filename=OUTPUT_DIR + 'session_data.json'):
    # ストリームリットのセッション状態を辞書形式に変換
    session_data = {key: value for key, value in st.session_state.items()}
    session_data = str(session_data).encode('utf-8').decode('utf-8').replace('\'', '\"').replace(': False', ': \"False\"').replace(': True', ': \"True\"').replace(': None', ':\"None\"')
    session_data = session_data.replace(': [(', ': \"[(\"").replace(")], ", ")]\", ')
    jsondata = json.loads(session_data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(jsondata, f, ensure_ascii=False, indent=4)

# セッションデータのインポート関数
#@handle_api_errors
def import_session_data_direct(filename=OUTPUT_DIR + 'session_data.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        st.session_state.update(json.load(f))

# セッションデータのインポート関数を改良
def import_session_data():
    uploaded_file = st.file_uploader('セッションデータをインポートするファイルを選択してください', type='json')
    if uploaded_file is not None:
        with open(uploaded_file.name, 'r') as f:
            st.session_state.update(json.load(f))
        st.success('セッションデータがインポートされました。')

