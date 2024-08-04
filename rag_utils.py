from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
import os

from dotenv import load_dotenv
load_dotenv(override=True)

# llm = ChatOpenAI(model="gpt-4o")

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.0)

# llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
# llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
# llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

# Debug: Print the Anthropic API key to verify it's loaded correctly
print("Anthropic API Key:", os.getenv("ANTHROPIC_API_KEY"))


def split_documents(documents):
    """
    Splits the given documents into smaller text chunks with a specified chunk size and overlap.
    
    Args:
        documents (List[str]): A list of text documents to be split.
    
    Returns:
        List[str]: A list of text chunks split from the input documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=300
    )
    texts = text_splitter.split_documents(documents)
    return texts

def create_db_and_retriever(texts):
    """
    Creates a Chroma database and retriever from the given text documents.
    
    Args:
        texts (List[str]): A list of text documents to create the database and retriever from.
    
    Returns:
        Retriever: A Chroma retriever that can be used to search the database of text documents.
    """
    # OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(
        search_type="mmr",  # use maximal marginal relevance to rank search results
        search_kwargs={"k": 20},
        )
    return retriever

def create_qa_chain(retriever):
    """
    Creates a question-answering (QA) chain that uses a retriever to find relevant context, and a language model to generate answers based on that context.
    
    The QA chain is composed of two sub-chains:
    1. A retriever chain that generates a search query based on the user's input and the conversation history.
    2. A document chain that generates an answer based on the retrieved context.
    
    The retriever chain uses a prompt that includes the conversation history and the user's input to generate a search query. The document chain uses a prompt that includes the retrieved context and the user's input to generate the answer.
    
    Args:
        retriever (Retriever): The retriever to use for finding relevant context.
    
    Returns:
        A QA chain that can be used to answer questions.
    """
    history_prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, history_prompt)

    current_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, current_prompt)

    qa_chain = create_retrieval_chain(retriever_chain, document_chain)
    
    return qa_chain

