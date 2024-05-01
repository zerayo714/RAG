import langchain
import openai
from langchain.llms import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
import streamlit as st

llm = OpenAI(
    api_key="YOUR_API_KEY",
    model_name="gpt-3.5-turbo"
)
st.title("我是檢索過RAG paper的聊天機器人")

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 留下歷史紀錄
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# 我的input
prompt = st.chat_input("Pass your prompt here")

@st.cache_resource
def create_vectorstore():
    pdf_name = "rag.pdf"  #待檢索的文件

    try:
        loaders = [PyPDFLoader(pdf_name)]
        #創建向量資料庫aka chromaDB
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20),
        ).from_loaders(loaders)
        return index
    
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

index = create_vectorstore()

if prompt:
    st.chat_message("user").markdown(prompt)  #我的prompt
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        response = llm(prompt)

        # 紀錄LLM回應並寫進dictionary
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        if index:
            #標準格式，缺一不可
            #type種類還有：
            # 1.map-reduce:檢索到的內容拆分多部份，分別丟入LLM最後再合併 -->處理大量複雜問題
            # 2.refine: 初步的內容丟入LLM，後續的內容加入迭代 -->須逐步完善的任務
            # 3.compact: 壓縮檢索到的內容再丟入LLM --> 需較大數據量or需要優化的情況
            chain = RetrievalQA.from_chain_type(    
                llm=llm,
                chain_type="stuff",  #把檢索到的所有內容傳遞給LLM
                retriever=index.vectorstore.as_retriever(),
                input_key="question",
            )

            result = chain({"question": prompt})

    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
