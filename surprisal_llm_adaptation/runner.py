import os
import gc
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import rdata
import wordfreq
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, get_scheduler
from torch.optim import AdamW
from datasets import Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, ttest_rel
from statsmodels.formula.api import mixedlm
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import seaborn as sns
from . utilities import CGLU_FILENAMES, assert_file_exists, combine_dfs, group_texts
from . surprisal_calc import SurprisalCalculator


class ExperimentRunner(object):
    """
    A class for running experiments on L2 English learner corpora using language models.
    This class handles the entire pipeline of:
    1. Processing and combining EFCAMDAT and CGLU learner corpora
    2. Adapting language models for different L1 backgrounds
    3. Estimating surprisal values
    4. Running regression analyses on reading times
    The pipeline processes data for 6 L1 languages:
    German, Italian, Mandarin, Portuguese, Russian, and Turkish.

    Attributes:
        model_id (str): The HuggingFace model ID for the base language model
        data_dir (str): Directory containing input corpora and where outputs will be saved
        l1s (list): List of L1 languages included in the analysis

    ## Methods:
        `get_efcamdat_dfs()`: Processes EFCAMDAT corpus data
        `get_cglu_dfs()`: Processes CGLU corpus data
        `combine_efcamdat_cglu()`: Combines processed corpus data
        `adapt_models()`: Adapts language models for each L1
        `evaluate_model_production()`: Compares adapted models with the baseline model
        `evaluate_model_representation()`: Evaluates model representation shift
        `get_meco_dfs()`: Processes MECO corpus data
        `get_regression_dfs()`: Prepares data for regression analysis
        `fit_regression_models()`: Runs regression analysis
        `graph_perplexities()`: Graphs model perplexities
        `graph_dlls()`: Graphs differences in log-likelihoods
        `run_experiment()`: Executes the complete experimental pipeline
    """

    def __init__(self, model_id: str = "EleutherAI/pythia-160m-deduped", data_dir: str = None):
        """
        Initializes the ExperimentRunner class.
        Args:
            model_id (str, optional): The HuggingFace ID of the language model to be used. Defaults to
                                      "EleutherAI/pythia-160m-deduped"
            data_dir (str, optional): The directory where the train and test corpora are located.
                                      This is also where data and model outputs will be saved.
                                      If None, defaults to the "data" directory in the current file's path.
        """

        self.model_id = model_id

        if not data_dir:
            self.data_dir = os.path.join('.', "data")
        else:
            self.data_dir = data_dir

        # Only L1s with eye-tracking data
        self.l1s = ["German", "Italian", "Mandarin", "Portuguese", "Russian", "Turkish"]

        # Private constants for internal use
        self._EFCAMDAT_COLS_OF_INTEREST = ["wordcount", "text", "l1"]
        self._CGLU_COLS_OF_INTEREST = ["N_Words", "Text"]
        self._MECO_COLS_OF_INTEREST = ["lang", "uniform_id", "trialid", "wordnum", "word", "firstrun.dur"]
        self._MECO_L1_CODES = ["ge", "it", "ch", "bp", "ru", "tr"]
        self.L1_COUNTRY_MAP = {
            "German": "Germany",
            "Italian": "Italy",
            "Mandarin": "Taiwan",
            "Portuguese": "Brazil",
            "Russian": "Russia",
            "Turkish": "Turkey"
        }


#   | ========================================================================================= |
#   |                                                                                           |
#   |                                   Private Helper Methods                                  |
#   |                                                                                           |
#   | ========================================================================================= |


    def _all_train_data_exists(self) -> bool:
        """
        Checks if all combined dfs already exist.
        Returns:
            bool: True if all training data exists, False otherwise.
        Raises:
            FileNotFoundError: If the combined training data folder exists but a dataframe is missing.
        """
        combined_train_folder = os.path.join(self.data_dir, "train_dfs", "combined")

        if os.path.exists(combined_train_folder):
            for l1 in self.l1s:
                assert_file_exists(os.path.join(combined_train_folder, f"{l1}_combined.feather"), 
                                         f"{l1} dataframe not found. " 
                                         f"Please remove conflicting folder {combined_train_folder} and rerun.")
            return True

        return False

    def _process_df_for_training(self,
                                 df: pd.DataFrame,
                                 tokenizer: AutoTokenizer,
                                 num_proc: int = 24,
                                 batch_size: int = 100,
                                 block_size: int = 2048
                                 ) -> tuple[Dataset, Dataset]:
        """
        Processes a DataFrame for training by tokenizing the text responses
        and creating train/dev Datasets with a 95/5 split.
        Args:
            df (pd.DataFrame): The DataFrame containing the data to be processed.
            tokenizer (AutoTokenizer): The tokenizer to use for tokenizing the text responses.
            num_proc (int, optional): The number of processes to use for parallelization. Default = 24
            batch_size (int, optional): The number of samples to be processed together during tokenization.
                                        Default = 100
            block_size (int, optional): The size of blocks to group texts into. Default = 2048
        Returns:
            tuple: A tuple containing (train_dataset, dev_dataset).
        """

        # Convert the DataFrame to a Dataset and remove wordcount column
        dataset = Dataset.from_pandas(df, preserve_index=False)
        dataset = dataset.remove_columns(["wordcount"])

        # Split: 95% train, 5% validation
        split_1 = dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset = split_1["train"]
        dev_dataset = split_1["test"]

        # Helpers to tokenize the data
        def tokenize_function(data):
            return tokenizer(data["text"])

        def tokenize(split):
            tokenized = split.map(tokenize_function,
                                batched=True,
                                num_proc=num_proc,
                                remove_columns=["text"])
            return tokenized.map(lambda x: group_texts(x, block_size),
                                batched=True,
                                batch_size=batch_size,
                                num_proc=num_proc)

        return tokenize(train_dataset), tokenize(dev_dataset)


    def _train_l1(self,
                  train_dataset: Dataset,
                  dev_dataset: Dataset,
                  l1: str,
                  output_dir: str,
                  per_device_train_batch_size: int = 16
                  ) -> None:
        """
        Adapts a (causal) language model on a specific dialect's Dataset.
        Args:
            train_dataset (Dataset): The dataset to train on.
            dev_dataset (Dataset): The dataset against which the model will be evaluated during training.
            l1 (str): The L1 language for which the model is being adapted.
            output_dir (str): Directory path where the new model will be saved.
            per_device_train_batch_size (int, optional): The batch size to use per GPU. Default = 16
        """

        # Constant training parameters
        EPOCHS = 4
        LEARNING_RATE = 0.00003
        WEIGHT_DECAY = 0.01

        # Set arguments for training
        training_args = TrainingArguments(output_dir=output_dir,
                                          num_train_epochs=EPOCHS,
                                          per_device_train_batch_size=per_device_train_batch_size,
                                          save_strategy="no",
                                          weight_decay=WEIGHT_DECAY,
                                          lr_scheduler_type="constant_with_warmup",
                                          learning_rate=LEARNING_RATE
                                          )
        
        # Initialize model, optimizer, and trainer
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("Warning: No GPU found. Using CPU instead, which may be slower for training.")

        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map=device)

        # For Apple Silicon computers
        # model = model.to("mps")

        # Set up optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        num_training_steps = len(train_dataset) * EPOCHS

        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=int(num_training_steps * .10),
            num_training_steps=num_training_steps
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            optimizers=(optimizer, scheduler)
        )

        # Train the model and save it
        trainer.train()
        trainer.save_model(os.path.join(output_dir, l1))

        # Free up memory
        del model
        del trainer
        del train_dataset
        del dev_dataset
        torch.cuda.empty_cache()
        gc.collect()


    def _average_meco_reading_times(self, l1: str) -> pd.DataFrame:
        """
        Calculates the average reading times for each word in the MECO data for a given L1.
        Returns the same DataFrame, but where a word's `firstrun.dur` is instead averaged across all
        participants, and individual participant reading times are dropped.
        Args:
            l1 (str): The L1 language for which to calculate average reading times.
        Returns:
            pd.DataFrame: A DataFrame containing the average reading times for each word.
        """

        # Load the MECO data for the given L1
        regression_df = pd.read_feather(os.path.join(self.data_dir, "test_dfs", "meco", f"{l1}_meco_clean.feather"))
        regression_df["firstrun.dur"] = regression_df["firstrun.dur"].fillna(0)

        # Group by word and calculate the average reading time
        regression_df = (
            regression_df.groupby(["trialid", "wordnum", "word"])["firstrun.dur"]
            .mean()
            .reset_index()
            .rename(columns={"firstrun.dur": "reading_time"})
            .sort_values(by=["trialid", "wordnum"])
        )

        return regression_df


    def _add_baseline_col(self, df: pd.DataFrame, calc: SurprisalCalculator):

        # Check if baseline surprisal has already been calculated
        baseline_surprisal_dir = os.path.join(self.data_dir, "temp_cache")
        baseline_surprisal_path = os.path.join(baseline_surprisal_dir, "baseline_surprisal.feather")
        if os.path.exists(baseline_surprisal_path):
            base_surp_df = pd.read_feather(baseline_surprisal_path)
            df["surp_base"] = base_surp_df["surp_base"]
            return df

        # Calculate surprisal
        df = calc.calculate_surprisal_dataframe(df, "baseline")

        if not os.path.exists(baseline_surprisal_dir):
            os.makedirs(baseline_surprisal_dir)

        # Save the dataframe to reuse baseline surprisal values
        df.to_feather(baseline_surprisal_path)
        return df


    def _add_t_minus_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add t-1 and t-2 columns for previous words' frequency, length, and surprisal (within each trial).
        """

        for lag in [1, 2]:
            df[f"freq_t-{lag}"] = (
                df.groupby("trialid")["freq"].shift(lag).fillna(0)
            )
            df[f"len_t-{lag}"] = (
                df.groupby("trialid")["len"].shift(lag).fillna(0)
            )
            df[f"surp_base_t-{lag}"] = (
                df.groupby("trialid")["surp_base"].shift(lag).fillna(0)
            )
            df[f"surp_adapt_t-{lag}"] = (
                df.groupby("trialid")["surp_adapt"].shift(lag).fillna(0)
            )
        return df


    def _train_eval_regression(self, regression_df: pd.DataFrame, base_cols: list[str], tgt_cols: list[str], y_col: str):
        """Performs cross-validated regression comparison between baseline and target models.
        This function implements a 10-fold cross-validation to compare two linear regression models:
        a baseline model using base_cols features and a target model using tgt_cols features.
        It computes the difference in log-likelihoods between the models and performs a
        paired permutation test to assess statistical significance.
        Args:
            regression_df (pd.DataFrame): DataFrame containing all required columns for regression
            base_cols (list[str]): Column name(s) for baseline model features
            tgt_cols (list[str]): Column name(s) for target model features
            y_col (str): Column name for dependent variable
        Returns:
            tuple: A tuple containing:
                - delta_log_likelihoods (list): A list of the delta log-likelihoods
                - avg_delta (float): Average difference in log-likelihoods between target and base models
                - p_value (float): P-value from paired permutation test (1000 permutations)
        Notes:
            - Uses 10-fold cross-validation with fixed random seed (42)
            - Assumes Gaussian noise model for log-likelihood calculations
            - Performs 1000 permutations for statistical testing
        """

        # Get y-values from `y_col` column
        y = regression_df[y_col].values

        # 10-fold cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        delta_log_likelihoods = []

        for train_idx, test_idx in kf.split(regression_df):
            # split
            X_base_train = regression_df.iloc[train_idx][base_cols].values
            X_base_test = regression_df.iloc[test_idx][base_cols].values
            X_tgt_train = regression_df.iloc[train_idx][tgt_cols].values
            X_tgt_test = regression_df.iloc[test_idx][tgt_cols].values
            y_train = y[train_idx]
            y_test = y[test_idx]

            # baseline regression
            base_model = LinearRegression().fit(X_base_train, y_train)
            mu_base = base_model.predict(X_base_test)
            sigma_base = np.std(y_train - base_model.predict(X_base_train))

            # target regression (with surprisal)
            tgt_model = LinearRegression().fit(X_tgt_train, y_train)
            mu_tgt = tgt_model.predict(X_tgt_test)
            sigma_tgt = np.std(y_train - tgt_model.predict(X_tgt_train))

            # log-likelihoods under Gaussian noise
            ll_base = norm.logpdf(y_test, loc=mu_base, scale=sigma_base)
            ll_tgt = norm.logpdf(y_test, loc=mu_tgt,  scale=sigma_tgt)
            delta_log_likelihoods.extend(ll_tgt - ll_base)

        # Compute average DLL
        delta_array = np.array(delta_log_likelihoods)
        avg_delta = delta_array.mean()

        # Paired permutation test
        PERMUTATIONS = 1000
        count = 0
        for i in range(PERMUTATIONS):
            signs = np.random.choice([-1, 1], size=delta_array.shape[0])
            if (signs * delta_array).mean() >= avg_delta:
                count += 1
        p_value = (count + 1) / (PERMUTATIONS + 1)

        return delta_log_likelihoods, avg_delta, p_value


#   | ========================================================================================= |
#   |                                                                                           |
#   |                                      Public Methods                                       |
#   |                                                                                           |
#   | ========================================================================================= |


    def get_efcamdat_dfs(self) -> None:
        """
        Retrieves and processes EFCAMDAT learner corpus data, splitting it by L1 language.
        This method reads the EFCAMDAT cleaned subcorpus Excel file, processes it by keeping only
        relevant columns, and splits the data into separate dataframes by L1 language.
        The resulting dataframes are saved as feather files.
        Raises:
            FileNotFoundError: If the EFCAMDAT cleaned subcorpus file or required L1 dataframes
                              are not found in the expected locations.
        """

        # Check if all necessary data already exists
        efcamdat_train_folder = os.path.join(self.data_dir, "train_dfs", "efcamdat")
        if os.path.exists(efcamdat_train_folder):
            for l1 in self.l1s:
                assert_file_exists(os.path.join(efcamdat_train_folder, f"{l1}_efcamdat.feather"), 
                                         f"{l1} dataframe not found. " 
                                         f"Please remove conflicting folder {efcamdat_train_folder} and rerun.")
            return

        # Check if the EFCAMDAT cleaned subcorpus exists, and read it into a DataFrame
        efcamdat_filename = "Final database (alternative prompts).xlsx"
        efcamdat_path = os.path.join(self.data_dir, "original_corpora", "efcamdat", efcamdat_filename)
        assert_file_exists(efcamdat_path,
                                f"Error: file {efcamdat_path} not found. "
                                "Please download the EFCAMDAT cleaned subcorpus from "
                                "https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html")

        train_df = pd.read_excel(efcamdat_path)

        # Drop all rows where `nationality` is `cn` to only get Taiwanese Mandarin speakers
        train_df = train_df[train_df["nationality"] != "cn"]

        # Drop all columns except the ones we need
        train_df = train_df[self._EFCAMDAT_COLS_OF_INTEREST]

        # Strip extraneous characters from the text
        train_df["text"] = train_df["text"].str.strip("\n\t ")

        # Split the dataframe by L1 for necessary L1s
        l1_dfs = {}
        for l1 in train_df["l1"].unique():
            if l1 in self.l1s:
                l1_dfs[l1] = train_df[train_df["l1"] == l1].reset_index(drop=True)

        # Make folder for L1 dataframes
        if not os.path.exists(efcamdat_train_folder):
            os.makedirs(efcamdat_train_folder)

        # Drop the L1 column and save the dataframes to feather files
        for l1 in l1_dfs:
            l1_dfs[l1] = l1_dfs[l1].drop(columns=["l1"])
            l1_dfs[l1].to_feather(os.path.join(efcamdat_train_folder, f"{l1}_efcamdat.feather"))


    def get_cglu_dfs(self) -> None:
        """
        Retrieves and processes CGLU data, This method reads the separated-by-country CGLU gzip files,
        process and combines them by language, and saves them as feather files.
        Raises:
            FileNotFoundError: If the EFCAMDAT cleaned subcorpus file or required L1 dataframes
                               are not found in the expected locations.
        """

        # Check if all necessary data already exists
        cglu_train_folder = os.path.join(self.data_dir, "train_dfs", "cglu")
        if os.path.exists(cglu_train_folder):
            for l1 in self.l1s:
                assert_file_exists(os.path.join(cglu_train_folder, f"{l1}_cglu.feather"), 
                                         f"{l1} dataframe not found. " 
                                         f"Please remove conflicting folder {cglu_train_folder} and rerun.")
            return
        
        # For each L1, check if the CGLU data exists, and read and save it
        cglu_dir_path = os.path.join(self.data_dir, "original_corpora", "cglu")
        for l1 in self.l1s:
            
            # Check if the CGLU data exists
            l1_cglu_paths = [os.path.join(cglu_dir_path, l1, filename) for filename in CGLU_FILENAMES[l1]]
            for path in l1_cglu_paths:
                assert_file_exists(path,
                                          f"Error: file {path} not found. "
                                          "Please download the CGLU data from "
                                          "https://publicdata.canterbury.ac.nz/Research/Geocorpus/CGLU_v5.2/")
                
            print(f"Getting CGLU data for {l1}...")
            
            # Read the CGLU data into a single DataFrame
            l1_cglu_dfs = [pd.read_csv(path, compression="gzip", index_col=0) for path in l1_cglu_paths]
            l1_cglu_df = pd.concat(l1_cglu_dfs, ignore_index=True)
            del l1_cglu_dfs
            gc.collect()
            
            # Drop all columns except the ones we need
            l1_cglu_df = l1_cglu_df[self._CGLU_COLS_OF_INTEREST]
            l1_cglu_df.rename(columns={"N_Words": "wordcount", "Text": "text"}, inplace=True)

            # Make folder for CGLU dataframe
            if not os.path.exists(cglu_train_folder):
                os.makedirs(cglu_train_folder)

            # Save the dataframe to a feather file
            l1_cglu_df.to_feather(os.path.join(cglu_train_folder, f"{l1}_cglu.feather"))
            del l1_cglu_df
            gc.collect()


    def combine_efcamdat_cglu(self) -> None:
        """Combines EFCAMDAT and CGLU training data for each L1 into single files.
        This method checks if combined training data already exists for each L1 language.
        If not, it reads the EFCAMDAT and CGLU data from separate files, combines them,
        and saves the combined data to new files.
        
        Assumes that EFCAMDAT and CGLU data has already been processed and saved using
        get_efcamdat_dfs() and get_cglu_dfs() respectively.

        Raises:
            AssertionError: If required input files are missing or if there are conflicts
                with existing combined data files.
        """

        # Check if all necessary data already exists
        combined_train_folder = os.path.join(self.data_dir, "train_dfs", "combined")
        if self._all_train_data_exists():
            print("All combined data already exists. Skipping combination step.")
            return


        # For each L1, check if the EFCAMDAT and CGLU data exists, and read and save it
        efcamdat_train_folder = os.path.join(self.data_dir, "train_dfs", "efcamdat")
        cglu_train_folder = os.path.join(self.data_dir, "train_dfs", "cglu")

        for l1 in self.l1s:

            # Check if the EFCAMDAT and CGLU data exist
            efcamdat_path = os.path.join(efcamdat_train_folder, f"{l1}_efcamdat.feather")
            cglu_path = os.path.join(cglu_train_folder, f"{l1}_cglu.feather")
            assert_file_exists(efcamdat_path,
                                      f"Error: file {efcamdat_path} not found. "
                                      "Please run get_efcamdat_dfs() first.")
            assert_file_exists(cglu_path,
                                      f"Error: file {cglu_path} not found. "
                                      "Please run get_cglu_dfs() first.")

            print(f"Combining {l1} EFCAMDAT and CGLU...")

            # Read the EFCAMDAT and CGLU data into DataFrames
            efcamdat_df = pd.read_feather(efcamdat_path)
            cglu_df = pd.read_feather(cglu_path)

            # Combine the two DataFrames
            combined_df = combine_dfs(efcamdat_df, cglu_df)
            del efcamdat_df, cglu_df
            gc.collect()

            # Make folder for L1 dataframes
            if not os.path.exists(combined_train_folder):
                os.makedirs(combined_train_folder)

            # Save the combined DataFrame to a feather file
            combined_df.to_feather(os.path.join(combined_train_folder, f"{l1}_combined.feather"))


    def adapt_models(self,
                     num_proc: int = 24,
                     batch_size: int = 100,
                     per_device_train_batch_size: int = 16,
                     block_size: int = 2048
                     ) -> None:
        """
        Adapt language models for each L1 in the dataset. For each L1, it:
        1. Loads the corresponding training data
        2. Processes it for training using the given model's tokenizer
        3. Adapts the model using that L1's data

        At the end, the adapted models are saved to `models` folder.

        Args:
            num_proc (int, optional): The number of processes (CPUs) to use in parallel during processing.
                                      Defaults to 24
            batch_size (int, optional): The number of samples to be processed together during tokenization.
                                        Defaults to 100
            per_device_train_batch_size (int, optional): The batch size to use per GPU.
                                                         Defaults to 16
            block_size (int, optional): The number of tokens in a single block to be given to the model.
                                        Defaults to 2048
        """

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Get the input and output directory for the adapted models
        input_dir = os.path.join(self.data_dir, "train_dfs", "combined")
        output_dir = os.path.join(self.data_dir, "models", self.model_id.split("/")[-1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for l1 in tqdm(self.l1s, desc="Adapting models", unit="models"):

            # Check if the model has already been adapted
            adapted_model_path = os.path.join(output_dir, l1)
            if os.path.exists(adapted_model_path):
                print(f"{l1} model found. Skipping...")
                continue

            print(f"\nAdapting {l1} model...")

            # Load the data for the current L1
            l1_df = pd.read_feather(os.path.join(input_dir, f"{l1}_combined.feather"))

            # Split the data into 95% train, 5% test
            train_df, test_df = train_test_split(l1_df, test_size=0.1, random_state=42)
            del l1_df

            # Make folder for held-out test data
            test_dfs_folder = os.path.join(self.data_dir, "test_dfs", "held_out_training")
            if not os.path.exists(test_dfs_folder):
                os.makedirs(test_dfs_folder)

            # Save the held-out dataframe
            test_df.to_feather(os.path.join(test_dfs_folder, f"{l1}_held_out.feather"))
            del test_df

            # Convert training df into a Dataset and process for training
            train_dataset, dev_dataset = self._process_df_for_training(train_df, 
                                                                       tokenizer,
                                                                       num_proc,
                                                                       batch_size,
                                                                       block_size)
            del train_df

            # Train the model and save it to `output_dir`
            self._train_l1(train_dataset, dev_dataset, l1, output_dir, per_device_train_batch_size)


    def evaluate_model_production(self):
        """
        For each L1, checks the statistical significance of if the adapted model's perplexity of
        the held-out data is less than that of the baseline (unadapted) model.
        Results are saved to `results/production` folder.
        """
        test_dfs_folder = os.path.join(self.data_dir, "test_dfs", "held_out_training")
        results_folder = os.path.join(self.data_dir, "results", "production")

        for l1 in self.l1s:

            results_path = os.path.join(results_folder, f"{l1}_production_results")
            if os.path.exists(results_path):
                print(f"{l1} production results found. Skipping...")
                continue

            print(f"\nEvaluating {l1} model production...")
            test_df = pd.read_feather(os.path.join(test_dfs_folder, f"{l1}_held_out.feather"))

            # Initialize tokenizer and surprisal calculator
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            calc = SurprisalCalculator(self.model_id, self.data_dir)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print("Warning: No GPU found. Using CPU instead, which may be slower for inference.")

            # Get baseline model
            base_model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map = device)

            # For Apple Silicon computers
            # base_model.to("mps")

            # Calculate baseline model perplexities
            base_perplexities = []
            for text in tqdm(test_df["text"]):
                base_ppl = calc.get_ppl(base_model, tokenizer, device, text)
                base_perplexities.append(base_ppl)

            # Free up memory
            del base_model
            gc.collect()

            # Get adapted model
            adapt_model_path = os.path.join(self.data_dir, "models", self.model_id.split("/")[-1], l1)
            adapt_model = AutoModelForCausalLM.from_pretrained(adapt_model_path, device_map = device)

            # For Apple Silicon computers
            # adapt_model.to("mps")

            # Calculate adapted model perplexities
            adapted_perplexities = []
            for text in tqdm(test_df["text"]):
                adapt_ppl = calc.get_ppl(adapt_model, tokenizer, device, text)
                adapted_perplexities.append(adapt_ppl)

            # Free up memory
            del adapt_model
            del tokenizer
            del calc
            gc.collect()

            # Run paired t-test to compare perplexities
            _, p_value = ttest_rel(base_perplexities, adapted_perplexities, alternative="greater")

            # Make folder for results
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            # Save results
            results_df = pd.DataFrame({
                'base_perplexities': base_perplexities,
                'adapted_perplexities': adapted_perplexities,
                'p_value': p_value
            })
            results_df.to_feather(results_path)

            # Print results
            print(f"\nProduction results for {l1}:")
            print(f"  Base perplexity mean: {np.mean(base_perplexities):.4f}")
            print(f"  Adapted perplexity mean: {np.mean(adapted_perplexities):.4f}")
            print(f"  p-value: {p_value:.8f}\n")


    def evaluate_model_representation(self):
        """
        For each L1, evaluates whether adaptation increases the model's divergence in perplexity
        between in-domain and out-domain texts, compared to the baseline model.
        Baseline perplexity for out-domain L1s is loaded from their pre-calculated production dataframes.
        Results are saved to `results/representation` folder.
        """

        # Define folders
        results_folder = os.path.join(self.data_dir, "results", "representation")
        test_dfs_folder = os.path.join(self.data_dir, "test_dfs", "held_out_training")
        production_folder = os.path.join(self.data_dir, "results", "production")

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        for l1 in self.l1s:

            result_path = os.path.join(results_folder, f"{l1}_representation_results.feather")
            summary_file_path = os.path.join(results_folder, f"{l1}_representation_results_summary.txt")

            if os.path.exists(result_path) and os.path.exists(summary_file_path):
                print(f"{l1} representation results and summary found. Skipping...")
                continue

            print(f"\nEvaluating representation shift for {l1}-adapted model...")

            # Load baseline and adapted perplexities for this L1
            l1_production_results_path = os.path.join(production_folder, f"{l1}_production_results")
            prod_df_l1 = pd.read_feather(l1_production_results_path)

            rows = []

            # Add precalculated in-domain perplexities to the rows list
            for i in range(len(prod_df_l1)):
                rows.append({
                    "model": "baseline", "domain": "in", 
                    "perplexity": prod_df_l1["base_perplexities"].iloc[i], 
                    "text_l1": l1
                })
                rows.append({
                    "model": "adapted", "domain": "in", 
                    "perplexity": prod_df_l1["adapted_perplexities"].iloc[i], 
                    "text_l1": l1
                })

            # Initialize tokenizer and surprisal calculator (for adapted model on out-domain texts)
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            calc = SurprisalCalculator(self.model_id, self.data_dir)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print("Warning: No GPU found. Using CPU for inference, which may be slower.")

            # Get the l1-adapted model
            adapted_model_path = os.path.join(self.data_dir, "models", self.model_id.split("/")[-1], l1)
            adapted_model = AutoModelForCausalLM.from_pretrained(adapted_model_path, device_map=device)

            # For Apple Silicon:
            # adapted_model.to("mps")

            # Process out-domain texts (texts from other_l1s)
            for other_l1 in self.l1s:
                if other_l1 == l1:
                    continue  # In-domain texts already processed

                # Load held-out texts for this other_l1
                other_l1_texts_path = os.path.join(test_dfs_folder, f"{other_l1}_held_out.feather")
                other_l1_texts_df = pd.read_feather(other_l1_texts_path)
                out_domain_texts_list = other_l1_texts_df["text"].tolist()

                # Load pre-calculated baseline perplexities for other_l1 texts
                other_l1_production_path = os.path.join(production_folder, f"{other_l1}_production_results")
                other_l1_prod_df = pd.read_feather(other_l1_production_path)
                precalculated_baseline_perplexities_on_other_l1 = other_l1_prod_df["base_perplexities"].tolist()

                # Calculate adapted model perplexities on these other_l1 texts
                adapted_model_perplexities_on_other_l1 = []
                desc_tqdm = f"PPL of {l1}-adapted model on {other_l1} texts"
                for text_content in tqdm(out_domain_texts_list, desc=desc_tqdm):
                    adapt_ppl = calc.get_ppl(adapted_model, tokenizer, device, text_content)
                    adapted_model_perplexities_on_other_l1.append(adapt_ppl)
                
                # Add results to rows
                for i in range(len(out_domain_texts_list)):
                    base_ppl = precalculated_baseline_perplexities_on_other_l1[i]
                    adapt_ppl_val = adapted_model_perplexities_on_other_l1[i]
                    
                    rows.append({
                        "model": "baseline", "domain": "out", 
                        "perplexity": base_ppl, 
                        "text_l1": other_l1
                    })
                    rows.append({
                        "model": "adapted", "domain": "out", 
                        "perplexity": adapt_ppl_val,
                        "text_l1": other_l1
                    })

            # Free up memory for the current l1 iteration
            del adapted_model
            del tokenizer 
            del calc
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            # Build DataFrame for statistical analysis
            df = pd.DataFrame(rows)

            # Save the DataFrame to a feather file
            df.to_feather(result_path)

            # Ensure 'text_l1' is suitable as a grouping variable (categorical)
            df["text_l1"] = df["text_l1"].astype("category")
            
            # Linear mixed-effects model
            lme_model = mixedlm("perplexity ~ model * domain", data=df, groups=df["text_l1"])
            result = lme_model.fit()
            print(result.summary())

            # Save the model summary
            with open(summary_file_path, "w") as f:
                f.write(result.summary().as_text())


    def get_meco_dfs(self) -> None:
        """
        Retreives and processes MECO data: wave 1 tracking measures, wave 2 tracking measures, and
        the MECO texts.

        This method reads the MECO data from the specified directory, processes it by keeping only
        relevant columns, and splits the data into separate dataframes by L1 dialect.
        The resulting dataframes are saved as feather files in the `test_dfs` directory.

        Raises:
            FileNotFoundError: If the MECO data directory or required L1 dataframes are not found
            ValueError: If there are participants present in both waves of the MECO data.
        """
        
        # Check if all necessary data already exists
        test_dfs_folder = os.path.join(self.data_dir, "test_dfs", "meco")
        if os.path.exists(test_dfs_folder):
            for l1 in self.l1s:
                assert_file_exists(os.path.join(test_dfs_folder, f"{l1}_meco_clean.feather"), 
                                   f"{l1} dataframe not found. " 
                                   f"Please remove conflicting folder {test_dfs_folder} and rerun.")
            assert_file_exists(os.path.join(test_dfs_folder, "texts.feather"),
                                   f"Texts dataframe not found. " 
                                   f"Please remove conflicting folder {test_dfs_folder} and rerun.")

            print("All MECO data already exists. Skipping data collection.")
            return
        
        # Check if the MECO data exists
        meco_dir_path = os.path.join(self.data_dir, "original_corpora", "meco", "release 2.0", "version 2.0")
        assert_file_exists(meco_dir_path,
                            f"Error: folder {meco_dir_path} not found. "
                            "Please download the release 2.0 MECO data from "
                            "https://osf.io/q9h43/")
        
        # Read the MECO data into a single DataFrame
        wave1_path = os.path.join(meco_dir_path, "wave 1", "primary data", "eye tracking data",
                                  "joint_data_l2_trimmed_version2.0.rda")
        wave2_path = os.path.join(meco_dir_path, "wave 2", "primary data", "eye tracking data",
                                  "joint_data_trimmed_L2_wave2_2025_01_03.rda")
        texts_path = os.path.join(meco_dir_path, "wave 1", "auxiliary files", "materials",
                                  "texts.meco.l2.rda")
        
        # Get the MECO eye-tracking data from both waves and the participant reading texts
        print("Reading MECO data...")

        wave1_data = rdata.read_rda(wave1_path)
        wave1_df = pd.DataFrame(wave1_data["joint.data"], columns=self._MECO_COLS_OF_INTEREST)

        wave2_data = rdata.read_rda(wave2_path)
        wave2_df = pd.DataFrame(wave2_data["joint.data"], columns=self._MECO_COLS_OF_INTEREST)

        texts_data = rdata.read_rda(texts_path)
        texts_df = pd.DataFrame(texts_data["d"])

        # Make folder for L1 dataframes
        if not os.path.exists(test_dfs_folder):
            os.makedirs(test_dfs_folder)

        # Save the texts dataframe
        texts_df.to_feather(os.path.join(test_dfs_folder, "texts.feather"))
        del texts_df
        
        # Check for overlap between participants (uniform_id) in both waves, if so raise an error
        wave1_participants = set(wave1_df["uniform_id"].unique())
        wave2_participants = set(wave2_df["uniform_id"].unique())
        overlap = wave1_participants.intersection(wave2_participants)
        if overlap:
            raise ValueError(f"Error: {len(overlap)} participants are present in both waves. "
                             "Please check the MECO data for duplicates.")

        # Concatenate the two DataFrames
        meco_df = pd.concat([wave1_df, wave2_df], ignore_index=True)
        del wave1_df, wave2_df

        # Take only the first portion of the language code (before the underscore)
        meco_df["lang"] = meco_df["lang"].str.split("_").str[0]

        # Drop all rows where `uniform_id` begins with `ch_s` to keep only Taiwanese Mandarin speakers
        meco_df = meco_df[~meco_df["uniform_id"].str.startswith("ch_s")]

        # Rename the language codes to the full language names
        meco_df["lang"] = meco_df["lang"].replace(dict(zip(self._MECO_L1_CODES, self.l1s)))

        # Save the full dataframe before splitting by dialect
        meco_df = meco_df[meco_df["lang"].isin(self.l1s)]
        meco_df.to_feather(os.path.join(test_dfs_folder, "full_meco_clean.feather"))

        # Fix mislabelled trialids
        fixes = {
            "it_25": {6: 7, 7: 8, 9: 10, 10: 11},
            "bp_23": {3: 4, 4: 5, 5: 6, 6: 7}
        }

        for uniform_id, fix_map in fixes.items():
            trialid_mask = meco_df["uniform_id"] == uniform_id

            # Sort the fixes in reverse order to handle later trial IDs first to avoid conflicts
            sorted_fixes = sorted(fix_map.items(), key=lambda x: x[0], reverse=True)
            for old_id, new_id in sorted_fixes:
                meco_df.loc[trialid_mask & (meco_df["trialid"] == old_id), "trialid"] = new_id

        # Group the data by L1 and save each L1's data to a separate DataFrame
        meco_dfs = {}
        for l1 in meco_df["lang"].unique():
            if l1 in self.l1s:
                meco_dfs[l1] = meco_df[meco_df["lang"] == l1].reset_index(drop=True)

        # Drop the lang column and save the dataframes to feather files
        for l1 in meco_dfs:
            meco_dfs[l1] = meco_dfs[l1].drop(columns=["lang"])
            meco_dfs[l1].to_feather(os.path.join(test_dfs_folder, f"{l1}_meco_clean.feather"))


    def get_regression_dfs(self) -> None:
        """
        Generate and save regression dataframes for each L1 language specified in self.l1s.
        For each L1, this method:
        1. Checks if regression data already exists (skips if found)
        2. Averages MECO reading times across participants
        3. Adds word frequency information (using negative log frequency)
        4. Adds character length information
        5. Adds surprisal values from both baseline and adapted models
        6. Adds t-minus columns (previous token information)
        7. Saves the resulting dataframe to a feather file
        The regression dataframes are stored in {data_dir}/test_dfs/regression/{l1}_regression.feather
        """

        # Initialize surprisal calculator (outside of loop for reuse)
        calc = SurprisalCalculator(self.model_id, self.data_dir)

        for l1 in self.l1s:

            # Check if the regression data exists
            regression_path = os.path.join(self.data_dir, "test_dfs", "regression", f"{l1}_regression.feather")
            if os.path.exists(regression_path):
                print(f"{l1} regression data found. Skipping...")
                continue

            print(f"\nGetting regression data for {l1}...")

            # Restructure the df to average reading times across participants
            regression_df = self._average_meco_reading_times(l1)

            # Add the word frequency column `freq`
            regression_df["freq"] = regression_df["word"].apply(lambda x: -wordfreq.word_frequency(x, "en"))

            # Add the character count column `len`
            regression_df["len"] = regression_df["word"].apply(lambda x: len(x.strip(".,")))

            # Add baseline and adapted model's surprisal columns `surp_base` and `surp_adapt`
            regression_df = self._add_baseline_col(regression_df, calc)
            regression_df = calc.calculate_surprisal_dataframe(regression_df, l1)

            # Add the `_t-n` columns
            regression_df = self._add_t_minus_col(regression_df)

            # Make folder for regression dataframes
            regression_folder = os.path.join(self.data_dir, "test_dfs", "regression")
            if not os.path.exists(regression_folder):
                os.makedirs(regression_folder)

            # Save the regression dataframe
            regression_df.to_feather(os.path.join(regression_folder, f"{l1}_regression.feather"))


    def fit_regression_models(self) -> None:
        """
        For each L1, outputs results to data_dir as results/{l1}_regression_results.feather
        """

        # Make folder for regression results
        regression_results_folder = os.path.join(self.data_dir, "results", "regression")
        if not os.path.exists(regression_results_folder):
            os.makedirs(regression_results_folder)

        for l1 in self.l1s:

            # Check if regression results already exist
            results_path = os.path.join(regression_results_folder, f"{l1}_regression_results.feather")
            if os.path.exists(results_path):
                print(f"{l1} regression results found. Skipping...")
                continue

            print(f"\nFitting regression models for {l1}...")

            # Get the regression DataFrame
            regression_path = os.path.join(self.data_dir, "test_dfs", "regression", f"{l1}_regression.feather")
            regression_df = pd.read_feather(regression_path)

            # Define predictors
            no_surprisal_cols = ["freq", "len", "freq_t-1", "len_t-1", "freq_t-2", "len_t-2"]
            baseline_cols = no_surprisal_cols + ["surp_base", "surp_base_t-1", "surp_base_t-2"]
            adapted_cols = no_surprisal_cols + ["surp_adapt", "surp_adapt_t-1", "surp_adapt_t-2"]

            nosurp_base_dlls = []
            nosurp_base_dll_avg = np.nan
            nosurp_base_p = np.nan
            base_adapt_dlls = []
            base_adapt_dll_avg = np.nan
            base_adapt_p = np.nan

            try:
                # Compare no surprisal vs baseline model surprisal
                nosurp_base_dlls, nosurp_base_dll_avg, nosurp_base_p = self._train_eval_regression(
                    regression_df,
                    base_cols=no_surprisal_cols, 
                    tgt_cols=baseline_cols,
                    y_col="reading_time"
                )
            except Exception as e:
                print("Error while training or evaluating no surprisal vs baseline surprisal "
                      f"{l1} model, setting delta and p-value to np.nan: {e}")

            try:
                # Compare baseline surprisal vs adapted surprisal
                base_adapt_dlls, base_adapt_dll_avg, base_adapt_p = self._train_eval_regression(
                    regression_df,
                    base_cols=baseline_cols,
                    tgt_cols=adapted_cols,
                    y_col="reading_time")
            except Exception as e:
                print("Error while training or evaluating baseline surprisal vs adapted surprisal "
                      f"{l1} model, setting delta and p-value to np.nan: {e}")
                
            try:
                # Compare no surprisal vs adapted model surprisal
                nosurp_adapt_dlls, nosurp_adapt_dll_avg, nosurp_adapt_p = self._train_eval_regression(
                    regression_df,
                    base_cols=no_surprisal_cols, 
                    tgt_cols=adapted_cols,
                    y_col="reading_time"
                )
            except Exception as e:
                print("Error while training or evaluating no surprisal vs adapted surprisal "
                      f"{l1} model, setting delta and p-value to np.nan: {e}")

            # Create a dataframe with comparison results
            results_df = pd.DataFrame({
                'pair': ["nosurp_base", "base_adapt", "nosurp_adapt"],
                'avg_dll': [nosurp_base_dll_avg, base_adapt_dll_avg, nosurp_adapt_dll_avg],
                'p-value': [nosurp_base_p, base_adapt_p, nosurp_adapt_p],
                'dlls': [nosurp_base_dlls, base_adapt_dlls, nosurp_adapt_dlls]
            })

            # Save results
            results_df.to_feather(results_path)

            # Print average dll and p-value for both pairs
            print(f"\nRegression results for {l1}:")
            print("No surprisal vs baseline surprisal:")
            print(f"  Average DLL: {nosurp_base_dll_avg:.4f}")
            print(f"  p-value: {nosurp_base_p:.8f}")
            print("\nBaseline surprisal vs adapted surprisal:")
            print(f"  Average DLL: {base_adapt_dll_avg:.4f}")
            print(f"  p-value: {base_adapt_p:.8f}\n")
            print("\nNo surprisal vs adapted surprisal:")
            print(f"  Average DLL: {nosurp_adapt_dll_avg:.4f}")
            print(f"  p-value: {nosurp_adapt_p:.8f}\n")


    def graph_perplexities(self):
        """
        Generates a single combined violin plot for all L1s, showing baseline and adapted
        perplexities. Outliers (values outside 2.5th-97.5th percentile) are removed
        per distribution per L1 before plotting. Violins are color-coded by L1.
        """
        results_folder = os.path.join(self.data_dir, "results", "production")
        graphs_folder = os.path.join(self.data_dir, "results", "graphs", "production_perplexities")

        if not os.path.exists(graphs_folder):
            os.makedirs(graphs_folder)

        all_l1_plot_data_list = []

        for l1_code in self.l1s:

            # Get country name from L1s
            country_name = self.L1_COUNTRY_MAP.get(l1_code, l1_code)

            # Read the production results for this L1
            results_file_name = f"{l1_code}_production_results"
            results_path = os.path.join(results_folder, results_file_name)
            results_df_original = pd.read_feather(results_path)

            plot_df_intermediate = results_df_original.copy()
            perplexity_columns = ['base_perplexities', 'adapted_perplexities']
            
            # Apply outlier removal for each perplexity column
            for col_name in perplexity_columns:
                if col_name in plot_df_intermediate and not plot_df_intermediate[col_name].dropna().empty:
                    valid_data = plot_df_intermediate[col_name].dropna()

                    lower_bound = valid_data.quantile(0.01)
                    upper_bound = valid_data.quantile(0.98)
                    
                    outliers_mask = (plot_df_intermediate[col_name] < lower_bound) | \
                                    (plot_df_intermediate[col_name] > upper_bound)
                    num_outliers_in_col = outliers_mask.sum()
                    if num_outliers_in_col > 0:
                        plot_df_intermediate.loc[outliers_mask, col_name] = np.nan

            l1_melted_data = pd.melt(plot_df_intermediate,
                                     value_vars=['base_perplexities', 'adapted_perplexities'],
                                     var_name='Model_Type_Raw',
                                     value_name='Perplexity')
            l1_melted_data['Country'] = country_name # Add L1 identifier
            l1_melted_data['Model_Type'] = l1_melted_data['Model_Type_Raw'].replace({
                'base_perplexities': 'Baseline',
                'adapted_perplexities': 'Adapted'
            })
            l1_melted_data = l1_melted_data.drop(columns=['Model_Type_Raw'])
            l1_melted_data = l1_melted_data.dropna(subset=['Perplexity'])

            all_l1_plot_data_list.append(l1_melted_data)

        master_plot_df = pd.concat(all_l1_plot_data_list, ignore_index=True)
        master_plot_df['Plot_Category'] = master_plot_df['Country'] + ' - ' + master_plot_df['Model_Type']
        
        # Determine the order of L1s and Model Types for the plot
        l1s_in_data_ordered = [self.L1_COUNTRY_MAP.get(l1_val, l1_val) for l1_val in self.l1s if self.L1_COUNTRY_MAP.get(l1_val, l1_val) in master_plot_df['Country'].unique()]
        model_type_order_plot = ['Baseline', 'Adapted']

        final_plot_category_order = []
        for l1_val in l1s_in_data_ordered:
            for mt_val in model_type_order_plot:
                category_name = f"{l1_val} - {mt_val}"
                final_plot_category_order.append(category_name)
            
        master_plot_df['Plot_Category'] = pd.Categorical(master_plot_df['Plot_Category'],
                                                         categories=final_plot_category_order,
                                                         ordered=True)

        if master_plot_df.empty or not final_plot_category_order:
            print("No data to plot after preparing categories. Cannot generate combined graph.")
            return
            
        num_l1s_plotted = len(l1s_in_data_ordered)
        fig_width = max(12, len(final_plot_category_order) * 1.1)
        fig_height = 8

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create customized palettes for different model types
        base_palette = sns.color_palette("tab10", n_colors=num_l1s_plotted)
        # Darken the colors for baseline models
        adapted_palette = [(r*0.85, g*0.85, b*0.85) for r, g, b in base_palette]

        # Create a dictionary mapping each Plot_Category to its color
        color_dict = {}
        for i, l1 in enumerate(l1s_in_data_ordered):
            # Assign colors based on model type within each L1
            color_dict[f"{l1} - Baseline"] = base_palette[i]
            color_dict[f"{l1} - Adapted"] = adapted_palette[i]
        
        # Use the color_dict to color the violins
        sns.violinplot(x='Plot_Category', y='Perplexity', 
                       data=master_plot_df,
                       order=final_plot_category_order,
                       palette=color_dict,
                       hue='Plot_Category',
                       dodge=False,
                       cut=0, inner="quartile", ax=ax, saturation=0.8, linewidth=1.0)

        ax.set_ylabel('Perplexity', fontsize=14, labelpad=12)
        ax.set_xlabel('Model', fontsize=14, labelpad=12)
        
        plt.xticks(rotation=30, ha="right", fontsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Create a custom legend
        handles = []
        labels = []
        for i, l1 in enumerate(l1s_in_data_ordered):
            # Use the base color for the legend
            patch = plt.Rectangle((0, 0), 1, 1, fc=base_palette[i])
            handles.append(patch)
            labels.append(l1)

        # Add the custom legend
        ax.legend(handles, labels, title='Country', loc='upper right', fontsize=12, title_fontsize=14,)

        plt.tight_layout(rect=[0, 0, 0.9, 1])

        combined_graph_filename = "perplexity_violin.png"
        combined_graph_path = os.path.join(graphs_folder, combined_graph_filename)

        plt.savefig(combined_graph_path, dpi=150)
        print(f"\nCombined graph saved to '{combined_graph_path}'")
        plt.close(fig)


    def graph_dlls(self):
        """
        Generates a single combined bar graph for all L1s, showing delta log-likelihoods (DLLs)
        for three model comparisons, with 95% confidence interval error bars.
        """
        regression_results_folder = os.path.join(self.data_dir, "results", "regression")
        graphs_folder = os.path.join(self.data_dir, "results", "graphs", "regression_dlls")

        if not os.path.exists(graphs_folder):
            os.makedirs(graphs_folder)
            print(f"Created directory for graphs: {graphs_folder}")

        all_l1_plot_data_list = []

        for l1_code in self.l1s:
            country_name = self.L1_COUNTRY_MAP.get(l1_code, l1_code)
            results_file_name = f"{l1_code}_regression_results.feather"
            results_path = os.path.join(regression_results_folder, results_file_name)
            results_df_original = pd.read_feather(results_path)

            results_df_original['Model_Comparison'] = results_df_original['pair'].replace({
                'base_adapt': 'Baseline vs. Adapted',
                'nosurp_adapt': 'No Surp. vs. Adapted',
                'nosurp_base': 'No Surp. vs. Baseline'
            })
            results_df_original['Country'] = country_name
            
            current_l1_data = results_df_original[['Country', 'Model_Comparison', 'dlls']].copy()

            exploded_df = current_l1_data.explode('dlls')
            exploded_df.rename(columns={'dlls': 'individual_dll'}, inplace=True)
            exploded_df['individual_dll'] = pd.to_numeric(exploded_df['individual_dll'], errors='coerce')
            exploded_df.dropna(subset=['individual_dll'], inplace=True)

            all_l1_plot_data_list.append(exploded_df)

        master_plot_df = pd.concat(all_l1_plot_data_list, ignore_index=True)

        model_comparison_order = [
            'No Surp. vs. Baseline',
            'No Surp. vs. Adapted',
            'Baseline vs. Adapted'
        ]
        # RGB multiplication factors: Lighter, Normal, Darker
        rgb_modification_factors = [1.25, 1.0, 0.75] 

        # Define hatches for the three model comparisons
        hatches = ['x', '-', '+']

        l1s_in_data_ordered = [self.L1_COUNTRY_MAP.get(l1_val, l1_val) for l1_val in self.l1s if self.L1_COUNTRY_MAP.get(l1_val, l1_val) in master_plot_df['Country'].unique()]

        master_plot_df['Country'] = pd.Categorical(master_plot_df['Country'], categories=l1s_in_data_ordered, ordered=True)
        master_plot_df['Model_Comparison'] = pd.Categorical(master_plot_df['Model_Comparison'], categories=model_comparison_order, ordered=True)
        
        master_plot_df = master_plot_df.dropna(subset=['individual_dll', 'Country', 'Model_Comparison'])
        master_plot_df = master_plot_df.sort_values(by=['Country', 'Model_Comparison'])

        if master_plot_df.empty:
            print("No data to plot after category preparation and NaN removal. Cannot generate graph.")
            return
            
        num_l1s = len(l1s_in_data_ordered)

        fig_width = max(8, num_l1s * 2.0)
        fig_height = 8.0 

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Ensure palette has at least 1 color, even if num_l1s might be 0 due to filtering
        base_l1_palette = sns.color_palette("tab10", n_colors=max(num_l1s, 1)) 
        country_to_base_color_map = {country: base_l1_palette[i] for i, country in enumerate(l1s_in_data_ordered)}
        
        # Create the barplot
        bar_plot = sns.barplot(x='Country', y='individual_dll', hue='Model_Comparison',
                    data=master_plot_df,
                    order=l1s_in_data_ordered,
                    hue_order=model_comparison_order,
                    ax=ax,
                    palette=['lightgray'] * len(model_comparison_order), 
                    errorbar=('ci', 95),
                    capsize=0.03,
                    width=0.7
                   )
        
        n_countries = len(l1s_in_data_ordered)
        
        for i, patch in enumerate(ax.patches):
            # Calculate country and comparison indices correctly
            country_index = i % n_countries  # Within each comparison group, get country index
            comparison_index = i // n_countries  # Which comparison group (0, 1, 2)
            
            if country_index < len(l1s_in_data_ordered):
                l1_name = l1s_in_data_ordered[country_index]
                base_color_rgb = country_to_base_color_map.get(l1_name)
                
                if base_color_rgb and comparison_index < len(rgb_modification_factors):
                    modification_factor = rgb_modification_factors[comparison_index]
                    final_color_r = min(1.0, max(0.0, base_color_rgb[0] * modification_factor))
                    final_color_g = min(1.0, max(0.0, base_color_rgb[1] * modification_factor))
                    final_color_b = min(1.0, max(0.0, base_color_rgb[2] * modification_factor))
                    final_color = (final_color_r, final_color_g, final_color_b)
                    patch.set_facecolor(final_color)
                    
                    # Add hatch pattern based on comparison type
                    if comparison_index < len(hatches):
                        patch.set_hatch(hatches[comparison_index])

        ax.set_ylabel('Average Δ Log-Likelihood', fontsize=18, labelpad=15)
        ax.set_xlabel('Country', fontsize=18, labelpad=15)
        ax.tick_params(axis='y', labelsize=14)
        plt.xticks(rotation=0, ha="right", fontsize=14)
        ax.axhline(0, color='grey', lw=1.0, linestyle='--')

        # Legend 1: Country colors legend
        current_sns_legend = ax.get_legend()
        if current_sns_legend is not None:
            current_sns_legend.remove()

        l1_handles = [plt.Rectangle((0,0),1,1, color=country_to_base_color_map[l1]) 
              for l1 in l1s_in_data_ordered if l1 in country_to_base_color_map]
        l1_labels = [l1 for l1 in l1s_in_data_ordered if l1 in country_to_base_color_map]

        # Legend 2: Model comparison legend with hatches
        hatch_handles = [plt.Rectangle((0,0),1,1, facecolor='gray', hatch=hatches[i]) 
             for i in range(len(model_comparison_order))]
        hatch_labels = model_comparison_order

        # Place the model comparison legend below the country legend, still inside the plot
        hatch_legend = ax.legend(hatch_handles, hatch_labels, 
              title='Model Comparison',
              loc='upper right',
              fontsize=12, title_fontsize=14, frameon=True,
              facecolor='white', framealpha=0.8)
        
        # Add the legend as an artist to maintain it
        ax.add_artist(hatch_legend)

        # Place the country legend inside the plot at the top right
        country_legend = ax.legend(l1_handles, l1_labels, 
                title='Country',
                loc='upper right',
                bbox_to_anchor=(1.0, 0.82),
                fontsize=12, title_fontsize=14, frameon=True, 
                facecolor='white', framealpha=0.8)
        
        # No need for special subplot adjustments since legends are inside
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
        
        # Set tight layout but respect the subplots_adjust settings
        fig.set_tight_layout(False)

        combined_graph_filename = "dll_bar_graph.png"
        combined_graph_path = os.path.join(graphs_folder, combined_graph_filename)

        plt.savefig(combined_graph_path, dpi=150, bbox_inches='tight')
        print(f"\nCombined DLL graph with CIs and hatches saved to '{combined_graph_path}'")
        plt.close(fig)


    def run_experiment(self,
                       num_proc: int = 24,
                       batch_size: int = 100,
                       per_device_train_batch_size: int = 16,
                       block_size: int = 2048
                       ) -> None:
        """
        Run the entire experiment pipeline automatically: get EFCAMDAT and CGLU dataframes,
        combine them, adapt models, gather data for regression, and fit the regression model,
        saving all artifacts (including the results of the experiment) in the `data_dir` directory.

        Args:
            num_proc (int, optional): The number of processes (CPUs) to use in parallel during processing.
                                      Defaults to 24
            batch_size (int, optional): The number of samples to be processed together during tokenization.
                                        Defaults to 100
            per_device_train_batch_size (int, optional): The batch size to use per GPU.
                                                         Defaults to 16
            block_size (int, optional): The number of tokens in a single block to be given to the model.
                                        Defaults to 2048
        """

        if self._all_train_data_exists():
            print("All training data already exists. Skipping data collection.")
        else:
            print("Collecting training data...")
            self.get_efcamdat_dfs()
            self.get_cglu_dfs()
            self.combine_efcamdat_cglu()

        self.adapt_models(num_proc, batch_size, per_device_train_batch_size, block_size)
        self.evaluate_model_production()
        self.evaluate_model_representation()

        self.get_meco_dfs()
        self.get_regression_dfs()
        self.fit_regression_models()
        self.graph_perplexities()
        self.graph_dlls()