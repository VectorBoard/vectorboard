import pandas as pd
from itertools import product
from rich.console import Console

import gradio as gr
import matplotlib.pyplot as plt

from vectorboard.experiment import Experiment


class GridSearch:
    """
    A class for performing grid search on a given chain.

    Attributes:
      experiments (list): A list of Experiment objects.
      experiment_results (DataFrame): A DataFrame containing the results of the experiments.
      console (Console): A Rich Console object.
      vectorstore_list (list): A list of vector stores.
      retrievers_list (list): A list of retrievers.
      chain (Chain): A Chain object.

    Methods:
      create_experiments(loader, param_grid): Creates a list of Experiment objects based on the given parameter grid.
      run(eval_queries): Runs the experiments on the given evaluation queries.
      results(gradio_app=True, share=False): Displays the results of the experiments in a Gradio app.
    """

    def __init__(self, chain):
        """
        Initializes a GridSearch object.

        Args:
          chain (Chain): A Chain object.
        """
        self.experiments = []
        self.experiment_results = None
        self.console = Console()
        self.vectorstore_list = []
        self.retrievers_list = []
        self.chain = chain

    def create_experiments(self, param_grid, loader=None, documents=None):
        """
        Creates a list of Experiment objects based on the given parameter grid.

        Args:
          param_grid (dict): A dictionary containing the parameter grid.
          loader (Loader): A Loader object.
          documents (list): A list of documents.

        Raises:
          ValueError: If the parameter grid does not contain 'embeddings', 'vector_store', or 'chunk_size'.
        """
        if "embeddings" not in param_grid.keys():
            raise ValueError("param_grid must contain embeddings")
        elif "vector_store" not in param_grid.keys():
            raise ValueError("param_grid must contain vector_store")
        elif "chunk_size" not in param_grid.keys():
            raise ValueError("param_grid must contain chunk_size")
        else:
            pass

        if documents is not None:
            self.documents = documents
        elif loader is not None:
            self.documents = loader.load()
        else:
            raise ValueError("Either documents or loader must be provided")

        all_params = [
            dict(zip(param_grid.keys(), values))
            for values in product(*param_grid.values())
        ]

        self.experiments_info_df = pd.DataFrame(
            all_params,
            index=[f"Experiment_{i}" for i in range(len(all_params))],
        )

        self.experiments = [
            Experiment(
                params=params,
                index=i + 1,  # to start from 1 instead of 0
                documents=self.documents,
                vectorstore_list=self.vectorstore_list,
                retrievers_list=self.retrievers_list,
                chain=self.chain,
            )
            for i, params in enumerate(all_params)
        ]

    def run(self, eval_queries):
        """
        Runs the experiments on the given evaluation queries.

        Args:
          eval_queries (list): A list of evaluation queries.
        """
        self.experiments_res_df = pd.DataFrame(index=eval_queries)

        for i, experiment in enumerate(self.experiments):
            response, runtime, embedding_time = experiment.run(eval_queries)

            self.experiments_res_df["Experiment_" + str(i)] = response.values()

            self.experiments_info_df.at["Experiment_" + str(i), "run time"] = runtime

            self.experiments_info_df.at[
                "Experiment_" + str(i), "embedding time"
            ] = embedding_time

    def results(self, gradio_app=True, share=False):
        """
        Displays the results of the experiments in a Gradio app.

        Args:
          gradio_app (bool): Whether to display the results in a Gradio app or not.
          share (bool): Whether to allow sharing of the Gradio app or not.
        """
        if gradio_app:

            def plot_fn():
                plt.figure(figsize=(50, 25))
                plt.style.use("seaborn-v0_8-poster")
                plt.style.use("seaborn-v0_8-deep")
                ax = self.experiments_info_df[["run time", "embedding time"]].plot.bar(
                    subplots=True, figsize=(15, 8), ylabel="Time (s)"
                )

                for axis in ax:
                    for p in axis.patches:
                        axis.annotate(
                            f"{p.get_height():.2f}",
                            (p.get_x() + p.get_width() / 2.0, p.get_height() / 2),
                            ha="center",
                            va="center",
                            color="white",
                            rotation=0,
                        )

                plt.xticks(rotation=30, horizontalalignment="center")
                plt.axhline(0, color="k")
                return plt

            with gr.Blocks(theme=gr.themes.Soft()) as demo:
                gr.Markdown(
                    """
            # 💛 Vectorboard Grid Search  
            ## Experiment results
          """
                )
                experiments = gr.Dataframe(self.experiments_info_df)
                queries = gr.Dataframe(
                    pd.DataFrame(self.experiments_res_df.index, columns=["Queries"])
                )
                result_table = gr.Dataframe(
                    self.experiments_res_df, show_label=True, label="Result table"
                )
                plot = gr.Plot(label="Plot")
                demo.load(plot_fn, [], [plot])

            demo.launch(share=True)
        else:
            pass
