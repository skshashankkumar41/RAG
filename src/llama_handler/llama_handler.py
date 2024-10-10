import logging
import qdrant_client
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.prompts.prompts import qa_prompt
from llama_index.llms.ollama import Ollama
import os
import shutil

logger = logging.getLogger(__name__)

class LlamaHandler():
    """
    A class to handle text and image embeddings, storage, and retrieval using Llama and Qdrant.

    Attributes:
        embed_model (HuggingFaceEmbedding): Embedding model for text.
        image_embed_model (ClipEmbedding): Embedding model for images.
        llm (Ollama): Language model for processing queries.
        mm_llm (OllamaMultiModal): Multi-modal language model for processing queries.
    """
    def __init__(self, text_embed_model='sentence-transformers/all-mpnet-base-v2', llm='mistral'):
        print(f"Loading Models...")
        self.embed_model = HuggingFaceEmbedding(text_embed_model,embed_batch_size=32)
        self.llm = Ollama(model=llm, request_timeout=500.0) if llm else None
        
    def _create_storage_context(self, text_store):
        """
        Create a storage context from the given text and image stores.

        Args:
            text_store (QdrantVectorStore): Vector store for text.
            image_store (QdrantVectorStore): Vector store for images.

        Returns:
            StorageContext: The storage context combining the text and image stores.
        """
        print(f"Creating storage context...")
        storage_context = StorageContext.from_defaults(vector_store=text_store)
        return storage_context 
    
    def _create_vector_stores(self, client, collection_name):   
        """
        Create a Qdrant vector store with the given client and collection name.

        Args:
            client (QdrantClient): The Qdrant client.
            collection_name (str): The name of the collection in Qdrant.

        Returns:
            QdrantVectorStore: The created vector store.
        """
        print(f"Creating vector store...")
        store = QdrantVectorStore(
                    client=client, collection_name=collection_name
                    )
        return store
    
    def _return_text_documents(self, pdf_path='./data/pdf', chunk_size=125, chunk_overlap=20):
        """
        Load and process text documents from a directory.

        Args:
            pdf_path (str): Path to the directory containing PDF files.
            chunk_size (int): Size of text chunks.
            chunk_overlap (int): Overlap between text chunks.

        Returns:
            list: List of processed text nodes.
        """
        print(f"Extracting Text from the pdf...")
        text_documents = SimpleDirectoryReader(pdf_path).load_data()
        node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(text_documents)
        return nodes
        
    def pipeline_v1_persist(self):
        """
        Create and persist a multi-modal index for text and images with different embedding(version 1).

        Returns:
            None
        """
        if os.path.exists('./data/index'):
            shutil.rmtree('./data/index')
        client = qdrant_client.QdrantClient(path="./data/index")
        text_store = self._create_vector_stores(client=client, collection_name='text_collection_v1')
        
        storage_context = self._create_storage_context(text_store)
        
        text_documents = self._return_text_documents()
        
        _ = VectorStoreIndex(text_documents, storage_context=storage_context, embed_model=self.embed_model)
        
        client.close()
        
        return None 
       
    def query_engine_v1(self, query, similarity_top_k=5):
        """
        Query the multi-modal index (version 1) and retrieve relevant text and image documents.

        Args:
            query (str): The query string.

        Returns:
            tuple: Retrieved texts and images.
        """
        client = qdrant_client.QdrantClient(path="./data/index")
        
        text_store = self._create_vector_stores(client=client, collection_name='text_collection_v1')
                        
        index = VectorStoreIndex.from_vector_store(
            vector_store=text_store,
            embed_model=self.embed_model
        )
        
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        retrieval_results = retriever.retrieve(query)
        
        
        retrieved_texts = []

        for doc in retrieval_results:
            if 'pdf' in doc.node.metadata.get('file_type'):
                print("DOC NODE:::::",doc.node.metadata)
                retrieved_texts.append({
                    'text':doc.node.text,
                    'file_path': doc.node.metadata.get('file_path'),
                    'page': doc.node.metadata.get('page_label')
                })
        
        print(f"Text Retrieval Results: {retrieved_texts}")
        
        client.close()
        
        return retrieved_texts
    
    
    def _create_context(self, retrieved_texts):
        """
        Create a context string from the retrieved text and image documents.

        Args:
            retrieved_texts (list): List of retrieved text documents.
            retrieved_images (list): List of retrieved image documents.

        Returns context
        """
        context = "Context: \n"
        context += "Textual contents: \n"
        for obj in retrieved_texts:
            context += obj['text'] + '\n\n'
            
        return context
    
    
    def answer_engine(self, question):
        """
        Answer a question by querying the multi-modal index and using the language model.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The answer generated by the language model.
        """
        retrieved_texts = self.query_engine_v1(question)
        context = self._create_context(retrieved_texts)
        
        prompt = qa_prompt.format(context_str=context, query_str=question)
        
        print(f"Prompt: {prompt}")
        
        response = self.llm.complete(prompt)
        
        print(f"Response: {response.text}")
        
        output = {
            'answer': response.text,
            'metadata':{
                'text':retrieved_texts
            }
        }
        
        return output
    

        
