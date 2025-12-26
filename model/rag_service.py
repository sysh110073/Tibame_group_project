import os
import time
import urllib.request
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
import configparser

# 讀取 Config
config = configparser.ConfigParser()
config.read('config.ini')

# 設定環境變數
os.environ["GOOGLE_API_KEY"] = config['GOOGLE']['GEMINI_API_KEY']
os.environ["PINECONE_API_KEY"] = config['PINECONE']['API_KEY']

# 全域變數
qa_chain = None

def init_rag_system():
    global qa_chain
    try:
        print("🚀 初始化 RAG 系統中...")
        
        # 1. 設定 Embeddings (使用 HuggingFace 開源模型，免費且效果不錯)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. 連線 Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index_name = "zero-waste-chef-recipes" 

        # 3. 檢查並建立 Index
        if index_name not in pc.list_indexes().names():
            print(f"📦 索引 {index_name} 不存在，正在建立中...")
            pc.create_index(
                name=index_name,
                dimension=384, # all-MiniLM-L6-v2 的維度是 384
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)

        index = pc.Index(index_name)
        
        # 4. 檢查是否需要下載並上傳資料 (Sharp 食譜)
        if index.describe_index_stats()['total_vector_count'] == 0:
            print("📥 雲端資料庫為空，開始下載 食譜 PDF...")
            
            pdf_filename = "sharp_recipes.pdf"

            # 如果檔案不在，才嘗試下載
            if not os.path.exists(pdf_filename):
                print("⚠️ 找不到本地 PDF，嘗試從網路下載...")
                # (使用 requests下載)
                import requests
                pdf_url = "https://tw.sharp/sites/default/files/products/documents/KN-V24AT%E9%A3%9F%E8%AD%9C%E9%9B%86.pdf"
                response = requests.get(pdf_url)
                if response.status_code == 200:
                    with open(pdf_filename, 'wb') as f:
                        f.write(response.content)
                    print("✅ 下載成功")
                else:
                    raise Exception("PDF 下載失敗，請手動下載 'sharp_recipes.pdf' 放到資料夾中")
            else:
                print("✅ 偵測到本地 PDF 檔案，直接使用。")
    

            print("📄 開始解析 PDF...")
            loader = PyPDFLoader(pdf_filename)
            print("📄 PDF 下載完成，開始解析與切割...")
            loader = PyPDFLoader(pdf_filename)
            docs = loader.load()
            
            # 切割文本：食譜通常比較短，chunk_size 設 500-800 效果較好
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            texts = text_splitter.split_documents(docs)
            
            print(f"🧩 共切割成 {len(texts)} 個片段，正在上傳至 Pinecone...")
            PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)
            print("✅ 資料上傳完畢！")
        
        # 5. 建立 Retriever 與 LLM
        vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings)
        # k=3 代表每次搜尋找回 3 個最相關的食譜片段
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 注意：目前 Google 穩定版是 gemini-1.5-flash，若無 2.5 權限請改回 1.5
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

        # 6. 設定 Prompt Template (角色扮演)
        custom_template = """
        你是「零剩食 AI 創意主廚」，致力於幫助使用者利用冰箱剩食做出美味料理。
        請根據下方的【參考食譜資料庫】（來源：Sharp 智慧鍋食譜）來回答用戶的問題。
        
        規則：
        1. 如果【參考食譜資料庫】中有合適的食譜，請優先參考其調味比例與烹飪時間。
        2. 如果資料庫中沒有完全匹配的菜色，請運用你身為 AI 主廚的知識進行創意改良，但請說明這是你的建議。
        3. 回答時請保持語氣親切、鼓勵環保。
        
        【參考食譜資料庫】：
        {context}
        
        用戶剩食/需求：{question}
        主廚建議回答：
        """
        PROMPT = PromptTemplate(template=custom_template, input_variables=["context", "question"])

        # 7. 建立 Chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        print("✅ AI 主廚系統 (RAG) 準備就緒！")

    except Exception as e:
        print(f"❌ RAG 初始化失敗: {e}")

# 透過這個函數給外部呼叫
def get_chef_response(user_input, chat_history=[]):
    if qa_chain is None:
        return "系統正在暖機中，請稍後再試..."
    
    # invoke 的輸入必須包含 question 與 chat_history
    result = qa_chain.invoke({"question": user_input, "chat_history": chat_history})
    return result["answer"]

# 在模組載入時自動執行初始化
init_rag_system()