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
        model_name="gpt-4o",
        temperature=0.4,
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(k=10),
    )

    res = qa.run(query)
    return res


from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/postMemory")
def post_memory():
    text = request.args.get("text")

    add_document_to_pinecone(text)

    return "success"


# System prompt to give some more guidance
PROMPT = """
You are an AI meant to help Alzheimer's patients remember their memories. An example question thhey might ask: "Tell me about a time I felt fulfilled."

They could ask about some more detail for a memory that they remember a little of.

Be kind and considerate.

USE AS MUCH DETAIL AS POSSIBLE. you want them to feel like they are living there again.

Respond in the second person.

Make it vivid and paraphrase. 

If you can't find any relevant memories, encourage them to go to the add memory page and have them or a family member add a memory.

REMEMBER THE INFORMATION THAT THE USER TELLS YOU TO, EVEN IF YOU CAN'T FIND ANY RELEVANT MEMORIES.

Do NOT
- mention anything about you being an AI.
- mention anything about context. 
- make up false information.


act like a human.

"""


@app.route("/query")
def hello_world():
    query = request.args.get("query")

    query = PROMPT + query

    print(query)

    res = get_llm_response(query)
    print(res)

    return res


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
