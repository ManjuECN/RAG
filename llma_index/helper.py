from llama_index.llms.llama_cpp import LlamaCPP
from  llama_index.llms.llama_cpp.llama_utils  import messages_to_prompt, completion_to_prompt
from llama_index.core import StorageContext, load_index_from_storage 
from llama_index.core import VectorStoreIndex # from llamaindex which is a vectorbase for documents
import os
import torch

def get_system_prompt():
    system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
    return system_prompt




def load_llm():

    system_prompt = get_system_prompt()
    llm = LlamaCPP(
        model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        temperature=0.1,
        max_new_tokens=256,
        context_window=4096,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": -1},
        system_prompt=system_prompt,
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm



def get_prompts():
    qa_prompt_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information in the pdf document, "
        "answer the questions only from the context provided: {query_str}\n"
    )

    refine_prompt_str = (
        "We have the opportunity to refine the original answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the question: {query_str}. "
        "If the context isn't useful, output the original answer again.\n"
        "Original Answer: {existing_answer}"
    )

    return qa_prompt_str, refine_prompt_str



def get_chat_text_qa_msgs():
    qa_prompt_str,refine_prompt_str = get_prompts()
    chat_text_qa_msgs = [
        (
            "system",
            "Always answer the question, from context provided.",
        ),
        ("user", qa_prompt_str),
    ]
    return chat_text_qa_msgs


def get_chat_refine_msgs():
    qa_prompt_str,refine_prompt_str = get_prompts()
    chat_refine_msgs = [
        (
            "system",
            "Always answer the question, from context provided.",
        ),
        ("user", refine_prompt_str),
    ]
    return chat_refine_msgs


# A function that helps to create Index, vectors and store them in a given folder.
def manage_index(documents, embed_model, node_parser, save_dir):
    # Check if the directory exists, and create or load the index accordingly
    if not os.path.exists(save_dir):
        # Create and load the index if it doesn't exist
        index = VectorStoreIndex.from_documents(
            [documents],
            embed_model=embed_model,  # Pass the embedding model
            node_parser=node_parser  # Pass the node parser
        )
        # Persist the index
        index.storage_context.persist(persist_dir=save_dir)
    else:
        # Load the existing index
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            embed_model=embed_model,  # Pass the embedding model
            node_parser=node_parser  # Pass the node parser
        )
    return index



def run_query(documents,embed_model,node_parser,save_dir,text_qa_template,refine_template,query):
    llm = load_llm()
    index = manage_index(documents, embed_model, node_parser, save_dir)
    return index.as_query_engine(
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        llm=llm,
    ).query(query)
