import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, get_scheduler
from torch.optim import AdamW
from datasets import Dataset
import torch
import pyreadr
import gc

from . utilities import CGLU_FILENAMES, assert_file_exists, combine_dfs, group_texts


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

    Methods:
        get_efcamdat_dfs(): Processes EFCAMDAT corpus data
        get_cglu_dfs(): Processes CGLU corpus data
        combine_efcamdat_cglu(): Combines processed corpus data
        adapt_models(): Adapts language models for each L1
        get_regression_df(): Prepares data for regression analysis
        fit_regression_model(): Runs regression analysis
        run_experiment(): Executes the complete experimental pipeline
    """

    def __init__(self, model_id: str = "EleutherAI/pythia-410m-deduped", data_dir: str = None):
        """
        Initializes the ExperimentRunner class.
        Args:
            model_id (str, optional): The HuggingFace ID of the language model to be used. Defaults to
                                      "EleutherAI/pythia-410m-deduped"
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

        # Minimum word count (for Turkish) to ensure equal amounts of data
        # self._MIN_WORDCOUNT = 126824


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
                                 ) -> Dataset:
        """
        Processes a DataFrame for training by tokenizing the text responses and creating a Dataset.
        Args:
            df (pd.DataFrame): The DataFrame containing the data to be processed.
            tokenizer (AutoTokenizer): The tokenizer to use for tokenizing the text responses.
            num_proc (int, optional): The number of processes to use for parallelization. Default = 24
            batch_size (int, optional): The number of samples to be processed together during tokenization.
                                        Default = 100
            block_size (int, optional): The size of blocks to group texts into. Default = 2048
        Returns:
            Dataset: A Dataset object containing the tokenized data.
        """

        def tokenize_function(data):
            return tokenizer(data["text"])

        # Convert the DataFrame to a Dataset and remove wordcount column
        dataset = Dataset.from_pandas(df, preserve_index=False)
        dataset = dataset.remove_columns(["wordcount"])

        # Tokenize the remaining text column
        tokenized_dataset = dataset.map(tokenize_function, 
                                        batched=True,
                                        num_proc=num_proc,
                                        remove_columns=["text"]
                                        )

        # Group the tokenized texts into blocks of the specified size
        lm_dataset = tokenized_dataset.map(
            lambda x: group_texts(x, block_size),
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc
        )

        return lm_dataset


    def _train_l1(self,
                  train_dataset: Dataset,
                  l1: str,
                  output_dir: str,
                  per_device_train_batch_size: int = 16
                  ) -> None:
        """
        Adapts a (causal) language model on a specific dialect's Dataset.
        Args:
            train_dataset (Dataset): The dataset to train on.
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
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map=device)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        num_training_steps = len(train_dataset) * EPOCHS

        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=int(num_training_steps * .10),
            num_training_steps=num_training_steps
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            optimizers=(optimizer, scheduler)
        )

        # Train the model and save it
        trainer.train()
        trainer.save_model(os.path.join(output_dir, l1))

        # Free up memory
        del model
        del trainer
        del train_dataset
        torch.cuda.empty_cache()
        gc.collect()


    def _calculate_perplexity(self, l1: str, text: str) -> float:
        """
        Calculates the perplexity of a given text using a pre-trained language model.
        Args:
            l1 (str): The L1 language for which the model is being used.
            text (str): The text to calculate perplexity for.
        Returns:
            float: The calculated perplexity of the text.
        """
        
        # Load the model and tokenizer (use base model if L1 is not in the list)
        if l1 in self.l1s:
            model = AutoModelForCausalLM.from_pretrained(os.path.join(self.data_dir, "models", l1))
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_id)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt")
        
        # Calculate loss and perplexity
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()


    # ========================================================================================= #


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
            
            # Read the CGLU data into a single DataFrame
            l1_cglu_dfs = [pd.read_csv(path, compression="gzip", index_col=0) for path in l1_cglu_paths]
            l1_cglu_df = pd.concat(l1_cglu_dfs, ignore_index=True)
            
            # Drop all columns except the ones we need
            l1_cglu_df = l1_cglu_df[self._CGLU_COLS_OF_INTEREST]
            l1_cglu_df = l1_cglu_df.rename(columns={"N_Words": "wordcount", "Text": "text"})

            # Make folder for L1 dataframes
            if not os.path.exists(cglu_train_folder):
                os.makedirs(cglu_train_folder)

            # Save the dataframes to feather files
            l1_cglu_df.to_feather(os.path.join(cglu_train_folder, f"{l1}_cglu.feather"))


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
            
            # Read the EFCAMDAT and CGLU data into DataFrames
            efcamdat_df = pd.read_feather(efcamdat_path)
            cglu_df = pd.read_feather(cglu_path)

            # Combine the two DataFrames
            combined_df = combine_dfs(efcamdat_df, cglu_df)
            del efcamdat_df, cglu_df

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

            # Convert into a Dataset and process for training
            train_dataset = self._process_df_for_training(l1_df, tokenizer, num_proc, batch_size, block_size)
            del l1_df

            # Train the model and save it to `output_dir`
            self._train_l1(train_dataset, l1, output_dir, per_device_train_batch_size)


    def get_regression_df(self) -> None:
        """
        Not yet implemented.
        """
        for l1 in self.l1s:
            # Collect average reading times (firstrun.dur) from MECO (joint_data_trimmed_L2_wave2_2025_01_03.rda)
            # Collect surprisal values from respective model
            # Calculate the negative log frequency (unigram surprisal)
            # Get the number of characters in the word
            pass


    def fit_regression_model(self) -> None:
        """
        Not yet implemented.
        """
        pass


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
        self.get_regression_df()
        self.fit_regression_model()