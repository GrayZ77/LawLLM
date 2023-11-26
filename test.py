from langchain.document_loaders import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
import asyncio
# llm, tokenizer = cli_demo.init_model()
embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese",
                                    model_kwargs={'device': "cuda:0"})

raw_documents = TextLoader("./test.txt", encoding="utf8").load()[0].page_content
headers_to_split_on = [
    ("##", "Header 2")
]
text_splitter = CharacterTextSplitter(chunk_size = 256, chunk_overlap = 0)
documents = text_splitter.split_text(raw_documents)

db = Chroma.from_texts(documents, embeddings)

def quest(query):
    embedding_vector = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(embedding_vector, k=3, fetch_k=10)
    for i, doc in enumerate(docs):
        print(f"{i+1}.", doc.page_content, "\n")

if __name__ == '__main__':
    query = "中国宪法第二条是什么"
    quest(query)

