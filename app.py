import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# CONFIGURAÇÃO
PASTA_ARQUIVOS = "meus_arquivos"
PASTA_BANCO = "banco_vetorial_chroma"
MODELO_LLM = "llama3"

st.set_page_config(page_title="Trabalho AIC", page_icon="🦙")
st.title("Chat confiável")

@st.cache_resource
def carregar_banco_de_dados():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(PASTA_BANCO) and os.listdir(PASTA_BANCO):
        print("Carregando banco existente...")
        vectorstore = Chroma(persist_directory=PASTA_BANCO, embedding_function=embeddings)
    else:
        print("Criando novo banco...")
        if not os.path.exists(PASTA_ARQUIVOS):
            os.makedirs(PASTA_ARQUIVOS)
            st.error(f"Criei a pasta '{PASTA_ARQUIVOS}'")
            return None

        documents = []
        
        # Carregar PDFs
        loader_pdf = DirectoryLoader(PASTA_ARQUIVOS, glob="*.pdf", loader_cls=PyPDFLoader)
        docs_pdf = loader_pdf.load()
        documents.extend(docs_pdf)
        print(f"PDFs carregados: {len(docs_pdf)}")

        # Carregar TXTs
        loader_txt = DirectoryLoader(PASTA_ARQUIVOS, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})
        docs_txt = loader_txt.load()
        documents.extend(docs_txt)
        print(f"TXTs carregados: {len(docs_txt)}")

        if not documents:
            st.warning("Nenhum arquivo PDF ou TXT encontrado na pasta.")
            return None

        # Dividir
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Salvar
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=PASTA_BANCO
        )
    
    return vectorstore


with st.spinner("Carregando modelo e arquivos..."):
    vectorstore = carregar_banco_de_dados()

if vectorstore:
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model=MODELO_LLM, temperature=0.1) 

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pergunte ao Llama 3..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um assistente que responde apenas com base no contexto abaixo. Responda em Português.\n\nContexto:\n{context}"),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        with st.chat_message("assistant"):
            with st.spinner(f"O {MODELO_LLM} está pensando..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                except Exception as e:
                    st.error(f"Erro: {e}")