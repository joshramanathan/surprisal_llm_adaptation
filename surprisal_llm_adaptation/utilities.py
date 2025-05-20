import os
import pandas as pd

# This is a dictionary of names of applicable CGLU files for each language.
# The keys are the language names, and the values are lists of filenames.
CGLU_FILENAMES = {
    "German": ["europe_west.Germany.eng.clean.IN.gz"],
    "Italian": ["europe_west.Italy.eng.clean.IN.gz"],
    "Mandarin": ["asia_east.Taiwan.eng.clean.IN.gz"],
    "Portuguese": ["america_brazil.Brazil.eng.clean.IN.gz"],
    "Russian": ["europe_russia.Russia.eng.1.IN.gz"],
    "Turkish": ["middle_east.Turkey.eng.clean.original.gz"]
}

def assert_file_exists(file_path: str, error_message: str):
    """
    Asserts that a file exists at the given path. If not, raises a FileNotFoundError with
    the provided error message.
    Args:
        file_path (str): The path to the file to check.
        error_message (str): The error message to raise if the file does not exist.
    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(error_message)


def combine_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Combines two DataFrames into a single DataFrame with 
    approximately equal amounts of `wordcount` of each.
    Args:
        df1 (pd.DataFrame): The first DataFrame to combine.
        df2 (pd.DataFrame): The second DataFrame to combine.
    Returns:
        pd.DataFrame: The combined DataFrame with duplicates removed.
    """

    # Calculate the average wordcount for each source to estimate the number of rows to sample
    avg_wordcount_df1 = df1["wordcount"].mean()
    avg_wordcount_df2 = df2["wordcount"].mean()
    min_wordcount = min(df1["wordcount"].sum(), df2["wordcount"].sum())
    num_rows_df1 = int(min_wordcount / avg_wordcount_df1)
    num_rows_df2 = int(min_wordcount / avg_wordcount_df2)

    # Ensure that the number of rows to sample is not greater than the available rows
    num_rows_df1 = min(num_rows_df1, len(df1))
    num_rows_df2 = min(num_rows_df2, len(df2))

    # Sample the required number of rows from each DataFrame and concatenate them
    df1_sampled = df1.sample(n=num_rows_df1, random_state=123)
    df2_sampled = df2.sample(n=num_rows_df2, random_state=456)
    combined_df = pd.concat([df1_sampled, df2_sampled], ignore_index=True)

    # Shuffle and return the new DataFrame
    combined_df = combined_df.sample(frac=1, random_state=789).reset_index(drop=True)
    return combined_df


def group_texts(examples, block_size: int):
    """
    Groups texts into blocks of a specified size.
    Args:
        examples (dict): A dictionary containing the texts to be grouped.
        block_size (int): The size of blocks to group texts into.
    Returns:
        dict: A dictionary containing the grouped texts.
    """
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    
    # Calculate the number of full blocks
    total_length = (total_length // block_size) * block_size
    
    # Split the concatenated texts into blocks
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    result["labels"] = result["input_ids"].copy()
    
    return result