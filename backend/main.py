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


def get_vectorstore():
    index_name = "memory-vault"
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
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

    res = qa.run(query)
    return res


if __name__ == "__main__":
    text = """
                my name is keshav kotamraju. i like to watch the stars.
            """

    add_document_to_pinecone(text)

    query = "tell me about keshav"

    res = get_llm_response(query)
    print(res)
