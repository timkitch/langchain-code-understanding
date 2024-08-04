from rag_utils import create_db_and_retriever, create_qa_chain, split_documents
from repo_utils import clone_repo, load_repo
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import collect_runs
from langsmith import Client

import uuid

from dotenv import load_dotenv
load_dotenv(override=True)

import datetime
import os

print(os.environ["LANGCHAIN_API_KEY"])

# Set the Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
if not os.environ["ANTHROPIC_API_KEY"]:
    raise ValueError("ANTHROPIC_API_KEY is not set in the environment variables")

client = Client()

tracer = LangChainTracer(client)

output_parser = StrOutputParser()

remote_repo_url = input("Enter repo URL: ")
branch = input("Enter branch name (press Enter for default branch): ").strip() or None
repo_path = clone_repo(remote_repo_url, branch)

documents = load_repo(repo_path)
texts = split_documents(documents)
retriever = create_db_and_retriever(texts)
qa_chain = create_qa_chain(retriever)


def get_save_preference():
    while True:
        save_chat = input("Do you want to save this chat to a file? (yes/no): ").lower()
        if save_chat in ['yes', 'no']:
            return save_chat == 'yes'
        print("Please enter 'yes' or 'no'.")

def main():
    # Simulate user. For Langsmith tracing only.
    user_id = str(uuid.uuid4())
    print(f"User ID: {user_id}")
    
    save_chat = get_save_preference()
    chat_file = None
    
    if save_chat:
        filename = input("Enter a filename to save the chat: ")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_filename = f"{filename}-{timestamp}.txt"
        chat_file = open(full_filename, 'w')
        print(f"Chat will be saved to: {full_filename}")
    
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
                
                answer = response["answer"]
                print("Answer:", answer + "\n")
                
                if chat_file:
                    chat_file.write(f"Question: {question}\n")
                    chat_file.write(f"Answer: {answer}\n\n")
                
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
    
    if chat_file:
        chat_file.close()
        print(f"Chat saved to {full_filename}")

if __name__ == "__main__":
    main()

