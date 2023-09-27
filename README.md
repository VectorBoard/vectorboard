# 💛 Vectorboard - `alpha 0.0.1`

[![Discord Follow](https://dcbadge.vercel.app/api/server/Ev48fWkbSu?style=flat)](https://discord.gg/Ev48fWkbSu)
[![Twitter Follow](https://img.shields.io/twitter/follow/vectorboard?style=social)](https://twitter.com/vectorboard)
[![Twitter Follow](https://img.shields.io/twitter/follow/thehamedmp?style=social)](https://twitter.com/thehamedmp)

## Embeddings Optimization and Eval Framework for RAG/LLM Applications

**Find the best hyperparameters for Embedding your data in RAG Pipeline**

![vectorboard banner image](docs/banner.png)

# TL;DR

1. Instal `vectorboard`

```bash
pip install vectorboard
```

2. Create a grid search of parameters you want to experiment. For example:

```python
param_grid = {
    "chunk_size": [50, 300, 500],
    "vector_store": [FAISS],
    "embeddings": [OpenAIEmbeddings(), HuggingFaceEmbeddings()],
}
```

3. Run the search using a `GridSearch()` (more search types upcoming).

```python
from vectorboard.search import GridSearch

# Create a GridSearch with the chain you'd like to try.
grid_search = GridSearch(chain=RetrievalQA)

# Use a document loader
grid_search.create_experiments(loader, param_grid=param_grid)
grid_search.run(eval_queries=eval_queries)
grid_search.results()
```

## Step by step overview of the example

Import `GridSearch()` from `vectorboard.search`

```python
from vectorboard.search import GridSearch
```

Create a `dict` with parameters and steps you want to search over.

```python
param_grid = {
    "chunk_size": [50, 300, 500],
    "vector_store": [FAISS],
    "embeddings": [OpenAIEmbeddings(), HuggingFaceEmbeddings()],
}
```

If the parameter is not one of the simple types (`int, str, ...`), you need to import the Class. E.g. to try different Embedding algorithms, first import them (we use `langchain` for simplicity):

```python
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
```

Initilize a Vectorboard object with the chain you want to run your experiment. Currently supporting `RetrivalQA`. More chains and custom chains are on the roadmap.

```python
search = GridSearch(chain=RetrievalQA)
```

Import a loader relevant to your data and include it as a parameter to `.create_experiments()`.

```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("recycling.pdf") # For example
search.create_experiments(param_grid=param_grid, loader=loader)
```

If you already loaded your data or have it available, use:

```python
search.create_experiments(param_grid=param_grid, documents=YOUR_DOCS)
```

Define eval queries and run the experiments:

```python
eval_queries = [
    "what percentage of waste is recyvled into materials in 2022?",
    # ...
]
search.run(eval_queries=eval_queries)
```

Finally, view the results in a Gradio app using `.results()` method. To get a publicly available link to share with your team, set the `share=True` paramater.

```python
search.results(share=True)
```

# Overview and Core concepts

RAG (Retreival Augmentation Generation) is great but it has a huge emphasis on perfect embeddings.

But it has challenges. It's hard to find the right

# Current status

- Built on top of [🦜⛓️Langchain]()
- Using `Gradio` for the final result page (with shareable links)

## Currently supported steps and parameters

1. Embeddings
1. Text and Document transformers
1. Vector Databases

# Roadmap

- [ ] Support more types of `Search`.
- [ ] Support more chains. `LLMChain` and custom chains in progress.
- [ ] Add `async` support to run `Experiments()` in parallel.
- [ ] TS/JS support.
- [ ] Add Eval tools and metrics.

Have a special feature request? Send your feedback/suggestion on our Discord community:
[![Discord Follow](https://dcbadge.vercel.app/api/server/Ev48fWkbSu?style=flat)](https://discord.gg/Ev48fWkbSu)
