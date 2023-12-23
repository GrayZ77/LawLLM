from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.prompts.chat import ChatMessagePromptTemplate
import os

embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese",
                                    model_kwargs={'device': "cuda:0"})

def load_documents(directory = "/root/DISC-LawLLM/法律文书"):
    print("loading documents……")
    raw_documents = DirectoryLoader(directory).load()
    text_splitter = CharacterTextSplitter(chunk_size = 256, chunk_overlap = 0)
    docs = text_splitter.split_documents(raw_documents)
    return docs

def store_chroma(docs, embeddings, persist_dirctory="VectorDataBase"):
    print("storing vectors……")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_dirctory)
    db.persist
    return db


def quest(query):
    embedding_vector = embeddings.embed_query(query)
    if not os.path.exists("/root/DISC-LawLLM/VectorDataBase"):
        documents = load_documents()
        db = store_chroma(documents, embeddings)
    else:
        db = Chroma(persist_directory="/root/DISC-LawLLM/VectorDataBase", embedding_function=embeddings)
    message = ''
    docs = db.similarity_search_by_vector(embedding_vector, k=3, fetch_k=10)
    for i, doc in enumerate(docs):
        message = message + f"{i+1}. {doc.page_content}  \n\n"
    
    return message

if __name__ == "__main__":
    print(quest(query = "中国人民有哪些权力"))


