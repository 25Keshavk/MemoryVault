import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

import uuid

# load env vars with dot-env
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()  # take environment variables from .env.

from langchain.docstore.document import Document

from urllib3 import make_headers
from urllib.parse import urlparse
from pinecone import Pinecone

from langchain.document_loaders import TextLoader


"""
From the pinecone source code
"""


def get_pinecone_index(client: Pinecone, index_name: str):
    indexes = client.list_indexes()
    index_names = [i.name for i in indexes.index_list["indexes"]]

    if index_name in index_names:
        index = client.Index(index_name)
    elif len(index_names) == 0:
        raise ValueError(
            "No active indexes found in your Pinecone project, "
            "are you sure you're using the right Pinecone API key and Environment? "
            "Please double check your Pinecone dashboard."
        )
    else:
        raise ValueError(
            f"Index '{index_name}' not found in your Pinecone project. "
            f"Did you mean one of the following indexes: {', '.join(index_names)}"
        )
    return index


def get_vectorstore():
    # Create VectorDB client
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    proxy_headers = None
    proxy_uri = os.getenv("PROXY")  # http://username:password@myproxy.com

    # If we have a proxy URI, set Pinecone to use it
    """
    Proxy code from https://github.com/langchain-ai/langchain/discussions/18763#discussioncomment-9446396
    """
    if proxy_uri:
        url = urlparse(proxy_uri)
        if url.username and url.password:
            proxy_headers = make_headers(
                proxy_basic_auth="%s:%s" % (url.username, url.password)
            )
    pinecone = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        proxy_url=proxy_uri,
        proxy_headers=proxy_headers,
    )

    index_name = "memory-vault"
    index = get_pinecone_index(pinecone, index_name)

    vectorstore = PineconeVectorStore(
        index,
        embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )

    return vectorstore


def add_document_to_pinecone(text: str):
    new_doc = Document(page_content=text)

    # do chunking for this new doc
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents([new_doc])

    # generate unique id for this doc that's shared across the chunks
    document_id = str(uuid.uuid4())
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id

    # get the vectorstore
    vectorstore = get_vectorstore()

    vectorstore.add_documents(chunks)


# the pinecone index stores "memories" = pieces of text
# this function uses an llm to query the index about memories
def get_llm_response(query: str):
    # get the vectorstore
    vectorstore = get_vectorstore()

    # completion llm
    llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name="gpt-3.5-turbo",
        temperature=0.0,
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    res = qa.invoke(query)
    return res


if __name__ == "__main__":
    text = """
                my name is keshav kotamraju. i like to watch the stars.
            """

    add_document_to_pinecone(text)

    query = "tell me about keshav"

    res = get_llm_response(query)
    print(res)
