from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def create_db_and_retriever(texts):
    embedding = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embedding)
    retriever = db.as_retriever()
    return retriever

def create_qa_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    Context: {context}

    Question: {input}

    Answer: Let's approach this step-by-step:""")

    stuff_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever, stuff_chain)

