import json
import os
import fnmatch
from tree_sitter import Parser
from tree_sitter_languages import get_language
from transformers import GPT2Tokenizer

def read_exclusion_patterns(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def generate_project_structure(root_dir, exclusion_file=None, include=None):
    """
    Generates the project structure using Tree-sitter for improved code parsing across multiple languages.
    
    Args:
        root_dir (str): The root directory of the project.
        exclusion_file (str): Path to the file containing exclusion patterns.
        include (str or list): Specifies which parts of the structure to include.
                               Can be 'directories', 'files', 'all_files', or a list of these.
    
    Returns:
        dict: A dictionary containing the project structure.
    """
    if exclusion_file:
        exclusion_patterns = read_exclusion_patterns(exclusion_file)
    else:
        exclusion_patterns = []
        
    # Normalize the include parameter
    if include is None:
        include = ['directories', 'files', 'all_files']
    elif isinstance(include, str):
        include = [include]
    
    # Validate include parameter
    valid_includes = {'directories', 'files', 'all_files'}
    if not set(include).issubset(valid_includes):
        raise ValueError(f"Invalid include option. Must be one or more of {valid_includes}")
        
    # Initialize Tree-sitter for multiple languages
    LANGUAGES = {
        '.py': get_language('python'),
        '.java': get_language('java'),
        '.c': get_language('c'),
        '.h': get_language('c'),
        '.yaml': get_language('yaml'),
        '.yml': get_language('yaml'),
        '.js': get_language('javascript'),
        '.json': get_language('javascript'),
        '.ts': get_language('typescript'),
        '.tsx': get_language('typescript'),
        '.jsx': get_language('javascript'),
        '.tf': get_language('hcl'),  # Terraform files use HCL (HashiCorp Configuration Language)
        '.hcl': get_language('hcl'),  # Also include .hcl files explicitly
        '.md': get_language('markdown'),  # Added Markdown support
        '.markdown': get_language('markdown'),  # Alternative extension for Markdown
        '.sh': get_language('bash'),  # Added Bash script support
        '.bash': get_language('bash'),  # Alternative extension for Bash scripts
        # Add more file extensions and languages as needed
    }


    project_structure = {
        "directories": [],
        "files": {},
        "all_files": [],
    }
    
    # Initialize the project structure based on the include parameter
    # project_structure = {
    #     'directories': [] if 'directories' in include else None,
    #     'files': {} if 'files' in include else None,
    #     'all_files': [] if 'all_files' in include else None
    # }
    
    def traverse_tree(node, file_info, lang):
        if lang in ['python', 'java']:
            if node.type == 'class_definition':
                class_name = node.child_by_field_name('name').text.decode('utf8')
                file_info['classes'].append(class_name)
            elif node.type in ['function_definition', 'method_declaration']:
                # Capture the full signature
                if lang == 'python':
                    name = node.child_by_field_name('name').text.decode('utf8')
                    parameters = node.child_by_field_name('parameters').text.decode('utf8')
                    signature = f"def {name}{parameters}:"
                elif lang == 'java':
                    return_type = node.child_by_field_name('type')
                    name = node.child_by_field_name('name').text.decode('utf8')
                    parameters = node.child_by_field_name('parameters').text.decode('utf8')
                    signature = f"{return_type.text.decode('utf8') if return_type else ''} {name}{parameters}"
                file_info['functions'].append(signature.strip())
                
        elif lang == 'c':
            if node.type == 'function_definition':
                signature = node.child_by_field_name('declarator').text.decode('utf8')
                file_info['functions'].append(signature)
            elif node.type == 'struct_specifier':
                struct_name_node = node.child_by_field_name('name')
                if struct_name_node and struct_name_node.text:
                    struct_name = struct_name_node.text.decode('utf8')
                    file_info['structs'].append(struct_name)
        elif lang in ['javascript', 'typescript']:
            if node.type == 'class_declaration':
                class_name = node.child_by_field_name('name').text.decode('utf8')
                file_info['classes'].append(class_name)
            elif node.type in ['function_declaration', 'method_definition']:
                signature = node.text.decode('utf8').split('\n')[0].strip()
                file_info['functions'].append(signature)
            elif node.type == 'arrow_function':
                if node.parent.type == 'variable_declarator':
                    name = node.parent.child_by_field_name('name').text.decode('utf8')
                    params = node.child_by_field_name('parameters').text.decode('utf8')
                    signature = f"{name} = {params} =>"
                    file_info['functions'].append(signature)
        elif lang == 'yaml':
            if node.type == 'block_mapping_pair':
                key = node.child_by_field_name('key').text.decode('utf8')
                file_info['keys'].append(key)
        elif lang == 'hcl':  # Handling for Terraform scripts
            if node.type == 'block':
                block_parts = []
                for child in node.named_children:
                    if child.type == 'identifier':
                        block_parts.append(child.text.decode('utf8'))
                
                block_signature = " ".join(block_parts)
                if not block_signature:
                    block_signature = f"unknown_block_{node.start_point[0]}"  # Use line number for uniqueness
                
                file_info['blocks'] = file_info.get('blocks', [])
                file_info['blocks'].append(block_signature)
            elif node.type == 'attribute':
                for child in node.named_children:
                    if child.type == 'identifier':
                        key = child.text.decode('utf8')
                        file_info['keys'] = file_info.get('keys', [])
                        file_info['keys'].append(key)
                        break
        elif lang == 'markdown':
            if node.type == 'atx_heading':
                atx_h_sequence = node.child_by_field_name('atx_h_sequence')
                inline = node.child_by_field_name('inline')
                if atx_h_sequence and inline:
                    heading_level = len(atx_h_sequence.text.decode('utf8').strip())
                    heading_text = inline.text.decode('utf8').strip()
                    file_info['headings'] = file_info.get('headings', [])
                    file_info['headings'].append(f"H{heading_level}: {heading_text}")
            elif node.type == 'fenced_code_block':
                language = node.child_by_field_name('info_string')
                if language:
                    language = language.text.decode('utf8').strip()
                else:
                    language = "unknown"
                file_info['code_blocks'] = file_info.get('code_blocks', [])
                file_info['code_blocks'].append(f"Code block ({language})")
            elif node.type == 'link':
                text = node.child_by_field_name('text')
                url = node.child_by_field_name('url')
                if text and url:
                    file_info['links'] = file_info.get('links', [])
                    file_info['links'].append(f"{text.text.decode('utf8')} -> {url.text.decode('utf8')}")
            elif lang == 'bash':
                if node.type == 'function_definition':
                    name = node.child_by_field_name('name').text.decode('utf8')
                    file_info['functions'] = file_info.get('functions', [])
                    file_info['functions'].append(name)
                elif node.type == 'variable_assignment':
                    name = node.child_by_field_name('name').text.decode('utf8')
                    file_info['variables'] = file_info.get('variables', [])
                    file_info['variables'].append(name)
                elif node.type == 'command':
                    command = node.text.decode('utf8').split('\n')[0].strip()
                    file_info['commands'] = file_info.get('commands', [])
                    file_info['commands'].append(command)


        for child in node.children:
            traverse_tree(child, file_info, lang)

            
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        # Modify dirnames in-place to exclude directories
        dirnames[:] = [d for d in dirnames if not any(fnmatch.fnmatch(d, pattern) for pattern in exclusion_patterns)]
        # project_structure["directories"].append(dirpath)
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            
            # Check if the file matches any exclusion pattern
            if not any(fnmatch.fnmatch(file_path, pattern) for pattern in exclusion_patterns):
                project_structure["all_files"].append(file_path)
                _, ext = os.path.splitext(filename)
                if ext in LANGUAGES:
                    with open(file_path, "rb") as file:
                        content = file.read()
                        parser = Parser()
                        parser.set_language(LANGUAGES[ext])
                        tree = parser.parse(content)
                        file_info = {
                            "classes": [],
                            "functions": [],
                            "structs": [],
                            "keys": [],
                        }
                        traverse_tree(tree.root_node, file_info, LANGUAGES[ext].name)
                        project_structure["files"][file_path] = file_info

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        # Modify dirnames in-place to exclude directories
        dirnames[:] = [d for d in dirnames if not any(fnmatch.fnmatch(d, pattern) for pattern in exclusion_patterns)]

        project_structure["directories"].append(dirpath)
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            project_structure["all_files"].append(file_path)
            _, ext = os.path.splitext(filename)
            if ext in LANGUAGES:
                with open(file_path, "rb") as file:
                    content = file.read()
                    parser = Parser()
                    parser.set_language(LANGUAGES[ext])
                    tree = parser.parse(content)
                    file_info = {
                        "classes": [],
                        "functions": [],
                        "structs": [],
                        "keys": [],
                    }
                    traverse_tree(tree.root_node, file_info, LANGUAGES[ext].name)
                    project_structure["files"][file_path] = file_info

    
    # Print the size of the project_structure dictionary
    print(f"Size of project_structure: {len(project_structure)}")
    
    # Estimate the number of tokens
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    project_structure_str = str(project_structure)
    tokens = tokenizer.encode(project_structure_str)
    print(f"Estimated number of tokens: {len(tokens)}")
    
    # Return only the specified parts of the project structure
    return {key: value for key, value in project_structure.items() if key in include}
    # return project_structure

def main():
    root_dir = input("Please enter the root directory of the project: ")
    while not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a valid directory.")
        root_dir = input("Please enter a valid root directory: ")
    
    exclusion_file = input("Please enter the path to the exclusion file (press Enter to skip): ").strip()
    
    # TODO: add better default exclusion handling
    exclusion_file = "./exclusions"
    
    # if exclusion_file:
        # while not os.path.isfile(exclusion_file):
        #     print(f"Error: {exclusion_file} is not a valid file.")
        #     exclusion_file = input("Please enter a valid exclusion file path (or press Enter to skip): ").strip()
        #     if not exclusion_file:
        #         break
    
    if exclusion_file:
        # project_structure = generate_project_structure(root_dir, exclusion_file)
        # project_structure = generate_project_structure(root_dir, exclusion_file, include=["directories"])
        # project_structure = generate_project_structure(root_dir, exclusion_file, include=["all_files"])
        project_structure = generate_project_structure(root_dir, exclusion_file, include=["files"])
    else:
        project_structure = generate_project_structure(root_dir)
    
    print("Project structure generated successfully.")
    
    # write the project structure to a file for later use
    with open('project-structure.json', 'w') as f:
        f.write(json.dumps(project_structure, indent=4))
        # files_structure = project_structure.get("files", {})
        # json.dump(project_structure, f, indent=4)
    f.close()
    
    # print(print_project_structure(project_structure))

if __name__ == "__main__":
    main()
