"""
This script loads a PDF document using PyPDFLoader, 
creates a GridSearch object from vectorboard.search,
and runs a grid search on the document using the RetrievalQA chain from langchain.chains.
The grid search uses the specified param_grid and eval_queries to find 
the best combination of parameters for the chain.
The results of the grid search are then shared using a Gradio app link.
"""

from vectorboard.search import GridSearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

loader = PyPDFLoader("recycling.pdf")

param_grid = {
    "chunk_size": [500],
    "vector_store": [FAISS],
    "embeddings": [OpenAIEmbeddings(), HuggingFaceEmbeddings()],
}
eval_queries = [
    "What is the trend in the amount of household waste collected and treated in 2022 compared to the previous year?",  # noqa: E501
    "what percentage of waste is recyvled into materials in 2022?",
    "what percentage of waste is recovered into energy in 2022?",
    "what is the total volume of waste treated for energy recovery per person?",
]


def main():
    vectorboard_grid_search = GridSearch(chain=RetrievalQA)
    vectorboard_grid_search.create_experiments(param_grid=param_grid, loader=loader)
    vectorboard_grid_search.run(eval_queries=eval_queries)
    vectorboard_grid_search.results(share=True)


if __name__ == "__main__":
    main()
