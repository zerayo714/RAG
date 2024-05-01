# RAG(Retrieval Augmentation Generation)

## 特性
- 檢索外部資料庫來提升Output品質和精準度
- 不需重新訓練LM
- 向量資料庫(ex.ChromaDB/WeaviateDB)
## 流程圖
- Indexing: 建置文本資料庫，分割和設置(Text embedding & Chunking & Overlapping)
- Retrieval: 檢索向量資料庫
- Generation: 生成回應

![](https://github.com/zerayo714/RAG/blob/main/%E7%A4%BA%E6%84%8F%E5%9C%96.jpg)
## Streamlit畫面
![](https://github.com/zerayo714/RAG/blob/main/Demo.png)

## 模型配置
For generation : ```gpt-3.5-turbo```  
For embedding : ```all-MiniLM-L12-v2```
## 套件使用
```python
import langchain
import openai
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
import streamlit as st
```

## 快速開始
```
streamlit run {filename}
```
