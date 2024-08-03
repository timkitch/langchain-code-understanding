import os
import re
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

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
