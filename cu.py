from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

from langchain_text_splitters import RecursiveCharacterTextSplitter

import re, os

from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_anthropic import ChatAnthropic

from langchain_groq import ChatGroq

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import collect_runs
from langsmith import Client

import uuid

from dotenv import load_dotenv
load_dotenv(override=True)

print(os.environ["LANGCHAIN_API_KEY"])

client = Client()

tracer = LangChainTracer(client)

output_parser = StrOutputParser()

# llm = ChatOpenAI(model="gpt-4")

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.0)

# llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
# llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
# llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")


# Clone
repo_path = "./repos"
if not os.path.exists(repo_path):
    os.makedirs(repo_path)

def clone_repo(remote_repo_url, branch=None):
    """
    Clones a remote Git repository to a local path.
    
    Args:
        remote_repo_url (str): The URL of the remote Git repository to clone.
        branch (str, optional): The specific branch to clone. If None, clones the default branch.
    
    Returns:
        str: The local path where the repository was cloned.
    """
    # Use a regular expression to extract the last part of the URL path
    match = re.search(r"([^/]+)\.git$", remote_repo_url)
    if match:
        repo_name = match.group(1)
    else:
        print("Could not extract repo name from URL")
        exit(1)
    
    local_repo_path = repo_path + "/" + repo_name
    
    # check if local_repo_path exists
    if os.path.exists(local_repo_path):
        print("Local repo path already exists. Not cloning.")
    else:
        print(f"Cloning repo {remote_repo_url} to {local_repo_path}...")
        if branch:
            print(f"Cloning branch: {branch}")
            Repo.clone_from(remote_repo_url, to_path=local_repo_path, branch=branch, depth=1)
        else:
            Repo.clone_from(remote_repo_url, to_path=local_repo_path)
        
    return local_repo_path
        
def load_repo(repo_path):
    """
    Loads a repository from the file system and returns the documents.
    
    Args:
        repo_path (str): The path to the repository on the file system.
    
    Returns:
        List[Document]: A list of documents loaded from the repository.
    """
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".java", ".py", ".go", ".c", ".cpp", ".h", ".cs", ".php", ".js"],
        # exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(parser_threshold=500),
        # NOTE: the parser may perform better for certain languages if the language is specified.
        # parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    documents = loader.load()
    return documents

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

remote_repo_url = input("Enter repo URL: ")
branch = input("Enter branch name (press Enter for default branch): ").strip() or None
repo_path = clone_repo(remote_repo_url, branch)

documents = load_repo(repo_path)
texts = split_documents(documents)
retriever = create_db_and_retriever(texts)
qa_chain = create_qa_chain(retriever)

# Simulate user. For Langsmith tracing only.
user_id = str(uuid.uuid4())
print(f"User ID: {user_id}")
        
stop = False
while not stop:
    
    try:
        question = input("Enter a question or task (or 'x' to exit): ")
        if (question == "x"):
            stop = True
            continue
    
        with collect_runs() as runs_cb:
            response = qa_chain.invoke({"input": question},
                        config = {"tags": ["project-type", "code-understanding"],
                        "metadata": {"user_id": user_id},
                        "run_name": "code-understanding-chain"
                        }
                        #  config= {"callbacks": [tracer]}
                    )
            
            print("Langsmith run id:", runs_cb.traced_runs[0].id)
            
            print("Answer:", response["answer"] + "\n")
            
            # Ratings could be any scale: 1-5, thumbs-up/down (1 or 0), number of stars, range of emojis... just have to translate to number or boolean.
            user_rating = int(input("Rate the response (1-5): "))
            user_comments = input("(Optional) Enter any comments for feedback: ")
            
            client.create_feedback(
                run_id=runs_cb.traced_runs[0].id,
                key="user-rating",
                score=user_rating,
                comment=user_comments
        )
    finally:
        wait_for_all_tracers()
    
    

