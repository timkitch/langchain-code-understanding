from repo_utils import clone_repo, load_repo
from rag_utils import split_documents, create_db_and_retriever, create_qa_chain
import os
from langchain_anthropic import ChatAnthropic
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

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.0)

remote_repo_url = input("Enter repo URL: ")
branch = input("Enter branch name (press Enter for default branch): ").strip() or None
repo_path = clone_repo(remote_repo_url, branch)

documents = load_repo(repo_path)
texts = split_documents(documents)
retriever = create_db_and_retriever(texts)
qa_chain = create_qa_chain(llm, retriever)

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
    
    

