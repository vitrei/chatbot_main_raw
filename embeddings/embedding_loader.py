from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

class EmbeddingLoader:
    def __init__(self):
        pass

    def load(self, embedding_name: str):
        if embedding_name == "intfloat/multilingual-e5-large-instruct":
            return HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large-instruct"
            )
        elif embedding_name == "intfloat/multilingual-e5-large":
            return SentenceTransformer('intfloat/multilingual-e5-large')
        else:
            return HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large-instruct"
            )