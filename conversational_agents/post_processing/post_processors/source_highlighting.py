from dataclasses import asdict
import json

import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from conversational_agents.post_processing.post_processors.base_post_processors import BasePostProcessor
from nltk.tokenize import sent_tokenize
from config import config


class SourceHighlighting(BasePostProcessor):

    def invoke(agent_state, llm_answer):

        if llm_answer.payload == None:
            llm_answer.payload = {}

        if llm_answer.rag_context == None or len(llm_answer.rag_context) == 0:
            highlight = {
                "key": llm_answer.content,
                "value": "unknown",
                "source_document": {
                    "text": None                                                   
                }
            }

            llm_answer.payload["source_highlight"] = { 
                "highlights": [highlight]
            }  

            return llm_answer

        language = 'english'
        language_code = config.get('application', 'language')
        if language_code.lower() == 'de':
            language = 'german'

        sentences = sent_tokenize(llm_answer.content, language=language)

        documents = [doc.content for doc in llm_answer.rag_context]

        embedding_request = {
            "priority": "high"
        }

        embeddings_service_url = "https://llm.opra-assistant.site/generate_embeddings" #TODO

        embedding_request['texts'] = documents
        response = requests.post(embeddings_service_url, headers={"Content-Type": "application/json"}, data=json.dumps(embedding_request)) 
        doc_embeddings = response.json()['embeddings']

        embedding_request['texts'] = sentences
        response = requests.post(embeddings_service_url, headers={"Content-Type": "application/json"}, data=json.dumps(embedding_request))
        sentence_embeddings = response.json()['embeddings']

        cosine_similarities = cosine_similarity(sentence_embeddings, doc_embeddings)
    	
        highlights = []

        for i, similarities in enumerate(cosine_similarities):
            most_similar_doc_index = np.argmax(similarities)
            most_similar_doc = documents[most_similar_doc_index]
            similarity_score = similarities[most_similar_doc_index] 

            highlight = {
                "key": sentences[i],
                "value": "unknown",
                "source_document": {
                    "text": most_similar_doc                                                   
                }
            }

            if similarity_score > 0.8:
                highlight['value'] = "moderate_trust"

            if similarity_score > 0.86:
                highlight['value'] = "high_trust"
            
            highlights.append(highlight)


        llm_answer.payload["source_highlight"] = { 
            "highlights": highlights
        }           

        return llm_answer