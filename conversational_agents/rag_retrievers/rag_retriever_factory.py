import os
import json

from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever

from embeddings.embedding_loader import EmbeddingLoader

from config import config


here = os.path.dirname(os.path.abspath(__file__))

class RAGRetrieverFactory:

    def __init__(self):
        configuration_file_name = config.get('conversational_agent_rag','rag_retriever_config_file')
        configuration_file = os.path.join(here, configuration_file_name)
        with open(configuration_file, 'r') as file:
            rag_retriever_configuration = json.load(file)

        self.retrievers = []

        embedding_loader = EmbeddingLoader()

        for retriever_config in rag_retriever_configuration['retrievers']:
            
            if retriever_config['type'] == 'Chroma':
                embeddings = embedding_loader.load(retriever_config['embedding_function'])

                vectorstore = Chroma(persist_directory=retriever_config['persist_directory'], embedding_function=embeddings)

                retriever = vectorstore.as_retriever(
                    search_type=retriever_config["search_type"],
                    search_kwargs={'score_threshold': retriever_config['score_threshold']}
                )
                
                self.retrievers.append(retriever)

        self.weights = rag_retriever_configuration['weights']

    def get_retrievers(self):
        return EnsembleRetriever(
            retrievers=self.retrievers, 
            weights=self.weights
        )  