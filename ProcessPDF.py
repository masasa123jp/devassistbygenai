import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import openai
from transformers import pipeline, RagTokenizer, RagRetriever, RagSequenceForGeneration, RagConfig
import os
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import pdfplumber
from datasets import Dataset
import faiss
import numpy as np
import torch

st.set_page_config(page_title="PDF解析ツール", layout="wide")

# PDFを読み込み、テキストを抽出する関数
def pdf_to_documents(pdf_path):
    documents = []
    # PDFを開く
    with pdfplumber.open(pdf_path) as pdf:
        # PDFの各ページをループ
        for i, page in enumerate(pdf.pages):
            # ページからテキストを抽出
            text = page.extract_text()
            if text:
                # 各ページのテキストとタイトルを辞書として追加
                documents.append({'text': text, 'title': f'Page {i + 1}'})
    return documents

# データセットとインデックスを準備する関数
def prepare_retriever(documents, dataset_path, index_path):
    try:
        # HuggingFace Datasetを作成
        dataset = Dataset.from_list(documents)
        
        # DPR Question Encoderのロード
        tokenizer_dpr = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        model_dpr = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        
        # ドキュメントのテキストを取得
        passages = dataset['text']
        
        # テキストをトークナイズ
        inputs = tokenizer_dpr(
            passages,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 埋め込みを生成
        with torch.no_grad():
            outputs = model_dpr(**inputs)
            embeddings = outputs.pooler_output.cpu().numpy()  # 'pooler_output' を使用
        
        # 埋め込みを正規化
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 埋め込みをリスト形式に変換
        embeddings_list = embeddings.tolist()
        
        # 'embeddings' カラムをデータセットに追加
        dataset = dataset.add_column('embeddings', embeddings_list)
        
        # データセットを保存
        dataset.save_to_disk(dataset_path)
        
        # FAISSインデックスの構築
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # 内積で検索
        index.add(embeddings)
        
        # インデックスを保存
        faiss.write_index(index, index_path)
        
        return tokenizer_dpr, model_dpr
    except Exception as e:
        print(f"Error in prepare_retriever: {e}")
        raise

# PDFのパスを指定して読み込み
pdf_path = r"C:/Users/user/OneDrive/DB/H30hDB/H201804DB_PM1_M000.pdf"
documents = pdf_to_documents(pdf_path)

# TesseractのWindows向け設定（Tesseractのパスを指定）
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# OpenAIのAPIキーを設定
openai.api_key = os.environ.get('OPENAI_API_KEY')

# データセットとインデックスのパスを指定（スペースなし）
dataset_path = r"D:/temp/custom_dataset"
index_path = r"D:/temp/custom_index.faiss"

# ディレクトリの存在確認と作成
if os.path.exists(dataset_path):
    if not os.path.isdir(dataset_path):
        raise ValueError(f"The path {dataset_path} exists and is not a directory.")
else:
    os.makedirs(dataset_path)

# データセットとインデックスを準備
tokenizer_dpr, model_dpr = prepare_retriever(documents, dataset_path, index_path)

# RagConfigのロード
config = RagConfig.from_pretrained("facebook/rag-sequence-nq")

# RagRetrieverの初期化
retriever_a = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    passages_path=dataset_path,  # 'dataset_path' を 'passages_path' に変更
    index_path=index_path,
    use_dummy_dataset=False,
    use_dummy_index=False,
    trust_remote_code=True,
)

# RAGモデルとトークナイザーのロード
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
# ジェネレータートークナイザーの pad_token を eos_token に設定
tokenizer.generator_tokenizer.pad_token = tokenizer.generator_tokenizer.eos_token

model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# リトリーバーをモデルに設定
model.set_retriever(retriever_a)

# RAGの設定
retriever_pipeline = pipeline(
    "rag-sequence",  # 適切なタスク名を使用
    model=model,
    tokenizer=tokenizer,
    retriever=retriever_a
)

# チャット履歴を保持するためのメモリ（コンテキストの保持）
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Streamlitアプリケーションの開始
st.title('Interactive Chat with OCR, AI & RAG')

# ファイルアップロード
uploaded_file = st.file_uploader("Choose a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

# PDFまたは画像ファイルを処理
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        st.write("Converting PDF to images...")
        # 一時的なファイルとして保存
        temp_pdf_path = "temp.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        images = convert_from_path(temp_pdf_path, poppler_path=r'C:/util2/poppler-24.07.0/Library/bin')  # Popplerのパスを指定
        for i, image in enumerate(images):
            st.image(image, caption=f'Page {i + 1}', use_column_width=True)
            text = pytesseract.image_to_string(image)
            st.write(f"Extracted text from page {i + 1}:")
            st.write(text)
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        text = pytesseract.image_to_string(image)
        st.write("Extracted text:")
        st.write(text)

    # チャットボックスのUI
    st.subheader("Chat with the extracted document")
    user_input = st.text_area("Ask a question:", key="input")

    if user_input:
        # 過去の質問履歴も文脈として含めて問い合わせを行う
        context = "\n".join(st.session_state['chat_history'])
        response = openai.chat.completions.create(
            engine="gpt-4o-mini",
            prompt=f"Context: {context}\nUser: {user_input}\nAI:",
        )
        answer = response['choices'][0].message.content.strip()

        # チャット履歴を更新
        st.session_state['chat_history'].append(f"User: {user_input}")
        st.session_state['chat_history'].append(f"AI: {answer}")

        # チャット履歴を表示
        st.write("Chat history:")
        for chat in st.session_state['chat_history']:
            st.write(chat)

        # 過去のチャット履歴に基づいてRAGの回答を生成
        rag_answer = retriever_pipeline(user_input)
        st.write("RAG-based Answer:")
        st.write(rag_answer[0]['generated_text'])
