from time import monotonic
from rich.console import Console

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI


class Experiment:
    """
    A class representing an experiment.

    Attributes:
      params (dict): A dictionary containing experiment parameters.
      index (int): An integer representing the index of the experiment.
      documents (list): A list of documents to be used in the experiment.
      vectorstore_list (list): A list of vector stores used in the experiment.
      retrievers_list (list): A list of retrievers used in the experiment.
      chain (object): An object representing the chain used in the experiment.
      query_results (dict): A dictionary containing the results of the queries.
      run_time (float): A float representing the time it took to run the experiment.
      embedding_time (float): A float representing the time it took to embed the documents.

    Methods:
      run(eval_queries): Runs the experiment with the given evaluation queries.
      _evaluate(chain, eval_queries): Evaluates the experiment with the given chain and evaluation queries.
    """

    def __init__(
        self,
        params,
        index=0,
        documents=None,
        vectorstore_list=None,
        retrievers_list=None,
        chain=None,
    ):
        """
        Initializes an Experiment object.

        Args:
          params (dict): A dictionary containing experiment parameters.
          index (int): An integer representing the index of the experiment.
          documents (list): A list of documents to be used in the experiment.
          vectorstore_list (list): A list of vector stores used in the experiment.
          retrievers_list (list): A list of retrievers used in the experiment.
          chain (object): An object representing the chain used in the experiment.
        """
        self.params = params
        self.index = index
        self.console = Console()
        self.documents = documents
        self.vectorstore_list = vectorstore_list
        self.retrievers_list = retrievers_list
        self.chain = chain

        self.query_results = {}

        # time variables
        self.run_time = 0
        self.embedding_time = 0

    def run(self, eval_queries):
        """
        Runs the experiment with the given evaluation queries.

        Args:
          eval_queries (list): A list of evaluation queries.

        Returns:
          A tuple containing the query results, the run time, and the embedding time.
        """
        self.console.log(f"Experiment {self.index} started")
        self.start_time = monotonic()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.params["chunk_size"], chunk_overlap=0
        )
        texts = text_splitter.split_documents(self.documents)

        self.pre_embedding = monotonic()
        db = self.params["vector_store"].from_documents(
            texts, self.params["embeddings"]
        )
        self.post_embedding = monotonic()
        self.vectorstore_list.append(db)

        retriever = db.as_retriever(search_kwargs={"k": 2})
        self.retrievers_list.append(retriever)

        qa = self.chain.from_chain_type(
            llm=OpenAI(), chain_type="stuff", retriever=retriever
        )

        self.pre_queries = monotonic()

        self._evaluate(qa, eval_queries)

        return self.query_results, self.run_time, self.embedding_time

    def _evaluate(self, chain, eval_queries):
        """
        Evaluates the experiment with the given chain and evaluation queries.

        Args:
          chain (object): An object representing the chain used in the experiment.
          eval_queries (list): A list of evaluation queries.
        """
        self.console.log(f"Evaluating experiment {self.index}")
        for q in eval_queries:
            res = chain.run(q)
            self.query_results[q] = res

        post_queries = monotonic()
        self.console.log(f"Finished experiment {self.index}")

        self.run_time = post_queries - self.start_time
        self.embedding_time = self.post_embedding - self.pre_embedding
