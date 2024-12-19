import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import openai
import io
import os
from pdf2image import convert_from_path

# ページ設定
st.set_page_config(page_title="PDF解析ツール", layout="wide")

# Popplerのパス設定
POPPLER_PATH = r'C:/util2/poppler-24.07.0/Library/bin'
# TesseractのWindows向け設定（Tesseractのパスを指定）
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# OpenAI APIキーの設定
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_type = 'openai'

# PDFの読み込みと画像・テキスト抽出
def extract_data_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text_data = ""
    images = []

    # 暗号化されている場合の復号化
    if reader.is_encrypted:
        try:
            reader.decrypt("")  # パスワードがある場合は指定
        except Exception as e:
            st.error(f"PDFの復号化に失敗しました: {e}")
            return text_data, images

    # PDFのページを走査してテキストと画像を抽出
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text_data += page.extract_text()

        # ページに含まれる画像の抽出処理
        if '/XObject' in page['/Resources']:
            xObject = page['/Resources']['/XObject'].get_object()
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                    data = xObject[obj]._data
                    image = Image.frombytes("RGB", size, data)
                    images.append(image)

    return text_data, images

# RAG技術による補完的な質問応答機能
def generate_response(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content.strip()

# OCRを使用して画像からテキストを抽出する関数
def extract_text_from_image(image):
    return pytesseract.image_to_string(image, lang='jpn')

# メインアプリ
st.title("PDFおよび画像解析ツール")

uploaded_file = st.file_uploader("PDFまたは画像ファイルをアップロードしてください", type=["pdf", "png", "jpg"])

system_message='あなたは、優秀なデータベースエンジニアで、またIPAのデータベーススペシャリスト資格の保有者です。'
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        st.write("PDFを画像に変換中...")

        # 一時的なPDFファイルを保存
        temp_pdf_path = "temp.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # PDFから画像に変換
        images = convert_from_path(temp_pdf_path, poppler_path=POPPLER_PATH)

        # 各ページの画像を表示してテキストを抽出
        for i, image in enumerate(images):
            st.image(image, caption=f'ページ {i + 1}', use_column_width=True)
            text = extract_text_from_image(image)
            st.write(f"ページ {i + 1} から抽出されたテキスト:")
            st.write(text)

        # 一時的なPDFファイルを削除
        os.remove(temp_pdf_path)

    else:
        # 画像ファイルの処理
        image = Image.open(uploaded_file)
        st.image(image, caption='アップロードされた画像', use_column_width=True)
        text = extract_text_from_image(image)
        st.write("抽出されたテキスト:")
        st.write(text)

    # オプションで、RAG技術を使用して生成したい質問やテキストを入力
    st.subheader("RAG技術による関連情報生成")
    prompt = st.text_area("生成したい内容を入力してください:")

    if st.button("生成"):
        response = generate_response(prompt)
        st.write("生成された応答:")
        st.write(response)
