import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from minicons import scorer

class SurprisalCalculator:
    def __init__(self, model_id: str = "EleutherAI/pythia-160m-deduped", data_dir: str = None):
        """
        Initializes the SurprisalCalculator.

        Args:
            model_id (str): Base model identifier for Hugging Face (e.g., "gpt2")
                            or a path to a base model. This is used for the tokenizer
                            and as a fallback if local adapted models aren't found.
            data_dir (str): Root directory where specific fine-tuned adapted models are stored.
                            The path to an adapted model will be constructed as:
                            data_dir/models/base_model_name_for_dir/l1_code/
        """

        self.model_id = model_id

        if not data_dir:
            self.data_dir = os.path.join('.', "data")
        else:
            self.data_dir = data_dir

        self._tokenizer_cache = {}
        self._hf_model_cache = {}
        self._minicons_scorer_cache = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def _get_tokenizer(self) -> AutoTokenizer:
        """
        Loads and caches the Hugging Face tokenizer.
        Note that adapted models use the tokenizer of their base model (self.model_id).
        """
        if self.model_id not in self._tokenizer_cache:
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._tokenizer_cache[self.model_id] = tokenizer
            except Exception as e:
                print(f"Error loading tokenizer for base model_id '{self.model_id}': {e}")
                raise
        return self._tokenizer_cache[self.model_id]


    def _get_hf_model(self, l1: str) -> AutoModelForCausalLM:
        """
        Loads and caches the Hugging Face model.
        It prioritizes loading a local L1-specific adapted model.
        If not found, or if loading fails, it falls back to the base model (self.model_id).
        """

        # Construct the path for the L1-specific local model
        base_model_name_for_dir = self.model_id.split("/")[-1]
        l1_model_path = os.path.join(self.data_dir, "models", base_model_name_for_dir, l1)

        model_to_attempt_loading = None
        attempted_local_l1_model = False

        if os.path.isdir(l1_model_path):
            # Local L1 model path exists, this is our primary candidate
            model_to_attempt_loading = l1_model_path
            attempted_local_l1_model = True
        else:
            # Local L1 model path does not exist, use the base model ID
            model_to_attempt_loading = self.model_id

        # The cache key is the actual identifier (path or Hub ID) being loaded
        cache_key = model_to_attempt_loading

        if cache_key not in self._hf_model_cache:
            # print(f"INFO: Model '{cache_key}' not in cache. Attempting to load...")
            try:
                model = AutoModelForCausalLM.from_pretrained(model_to_attempt_loading)
                model.to(self.device)
                model.eval()
                self._hf_model_cache[cache_key] = model
                # print(f"INFO: Successfully loaded and cached model '{cache_key}'.")
            except Exception as e:
                print(f"Warning: Error loading model '{model_to_attempt_loading}': {e}")
                # If loading the local L1 model path failed, and it's different from the base model_id,
                # then explicitly try to load the base model_id as a fallback.
                if attempted_local_l1_model and model_to_attempt_loading != self.model_id:
                    print(f"INFO: Fallback: Attempting to load base model '{self.model_id}'.")
                    # Check cache for base model ID first
                    if self.model_id not in self._hf_model_cache:
                        try:
                            base_model_instance = AutoModelForCausalLM.from_pretrained(self.model_id)
                            base_model_instance.to(self.device)
                            base_model_instance.eval()
                            self._hf_model_cache[self.model_id] = base_model_instance
                            # print(f"INFO: Successfully loaded and cached base model '{self.model_id}' during fallback.")
                        except Exception as base_e:
                            print(f"Error: Failed to load base model '{self.model_id}' during fallback: {base_e}")
                            raise base_e # Re-raise error from loading base model
                    return self._hf_model_cache[self.model_id] # Return base model from cache
                # If it was already attempting to load self.model_id (as fallback or primary) and failed, re-raise.
                raise e
        
        return self._hf_model_cache[cache_key]


    def _get_minicons_scorer(self, l1: str) -> scorer.IncrementalLMScorer:
        """
        Creates and caches a minicons IncrementalLMScorer for the given language,
        using the appropriate (potentially local L1-specific) Hugging Face model.
        """
        # Cache key for minicons scorer should be specific to the l1 model context
        scorer_cache_key = f"minicons_{self.model_id}_{l1}"
        if scorer_cache_key not in self._minicons_scorer_cache:
            hf_model_instance = self._get_hf_model(l1) # Gets potentially L1-specific HF model
            tokenizer_instance = self._get_tokenizer() # Uses base tokenizer
            
            ilm_scorer = scorer.IncrementalLMScorer(hf_model_instance, tokenizer=tokenizer_instance, device=self.device)
            self._minicons_scorer_cache[scorer_cache_key] = ilm_scorer
        return self._minicons_scorer_cache[scorer_cache_key]


    def _calculate_surprisal_for_trial(self, trial_df: pd.DataFrame, l1: str) -> pd.Series:
        ilm_model = self._get_minicons_scorer(l1)
        hf_tokenizer = self._get_tokenizer() 

        original_words_from_df = trial_df['word'].astype(str).tolist()
        trial_surprisals = [np.nan] * len(original_words_from_df)

        # Prepare list of non-empty words and their original indices
        valid_words_info = []
        for i, word_str in enumerate(original_words_from_df):
            cleaned_word = word_str.strip()
            valid_words_info.append((i, cleaned_word))

        # Join (non-empty, stripped) words for processing by minicons
        trial_text = " ".join([info[1] for info in valid_words_info])

        # Get the surprisal scores for the entire trial text
        token_surprisal_data = ilm_model.token_score([trial_text], surprisal=True, base_two=True)

        # token_surprisal_data is a list of tuples (subtoken_str, surprisal_val)
        all_subtokens_for_trial = token_surprisal_data[0]

        # Tracks the next subtoken in all_subtokens_for_trial to be consumed
        global_subtoken_cursor = 0

        # Iterate through the valid words that need their surprisals calculated
        for word_info_idx, (original_df_idx, target_word_str) in enumerate(valid_words_info):
            current_word_accumulated_surprisal = 0.0
            matched_this_word = False

            # Try to build the current target_word_str using subtokens starting from the global_subtoken_cursor
            start_cursor_for_this_word_attempt = global_subtoken_cursor
            
            # Find the shortest span of subtokens that matches target_word_str
            for k_lookahead_idx in range(start_cursor_for_this_word_attempt, len(all_subtokens_for_trial)):

                # Consider subtokens from start_cursor_for_this_word_attempt up to k_lookahead_idx (inclusive)
                candidate_subtoken_span_strings = [
                    st[0] for st in all_subtokens_for_trial[start_cursor_for_this_word_attempt : k_lookahead_idx + 1]
                ]
                
                reconstructed_segment = hf_tokenizer.convert_tokens_to_string(candidate_subtoken_span_strings)
                reconstructed_segment_cleaned = reconstructed_segment.strip()

                if reconstructed_segment_cleaned == target_word_str:
                    # Exact match found
                    current_word_accumulated_surprisal = sum(
                        st[1] for st in all_subtokens_for_trial[start_cursor_for_this_word_attempt : k_lookahead_idx + 1]
                    )
                    trial_surprisals[original_df_idx] = current_word_accumulated_surprisal

                    # Advance global cursor past these consumed tokens
                    global_subtoken_cursor = k_lookahead_idx + 1
                    matched_this_word = True

                    # Move to the next target_word_str
                    break

        return pd.Series(trial_surprisals, index=trial_df.index)


    def calculate_surprisal_dataframe(self, df: pd.DataFrame, l1: str) -> pd.DataFrame:
        """
        Calculates word-level surprisal for each word in the input DataFrame using minicons.

        Args:
            df (pd.DataFrame): DataFrame with columns `trialid, wordnum, word`.
            l1 (str): The L1 language code for model selection.

        Returns:
            pd.DataFrame: The input DataFrame with an added column `surp`
                          containing surprisal values in bits.
        """

        # Assume column name from l1 input
        col_name = "surp_adapt"
        if l1 == "base" or l1 == "baseline":
            col_name = "surp_base"

        # Sort, just in case the DataFrame isn't sorted already
        df_sorted = df.sort_values(by=["trialid", "wordnum"])

        # Store series to concat later
        all_surprisals_list = []
        
        # Determine group keys to iterate over
        group_keys = df_sorted["trialid"].unique()

        for key in group_keys:
            group_df = df_sorted[df_sorted['trialid'] == key]
            if group_df.empty:
                continue
            temp_df_for_trial = pd.DataFrame({
                'word': group_df['word'],
                'trialid': group_df['trialid'] # Pass trialid for debug prints
            }, index=group_df.index)

            trial_surprisals_series = self._calculate_surprisal_for_trial(temp_df_for_trial, l1)
            all_surprisals_list.append(trial_surprisals_series)

        # Concatenate all the series into a single Series
        surprisal_column_concat = pd.concat(all_surprisals_list)

        # Assign back to the sorted DataFrame.
        df_sorted[col_name] = surprisal_column_concat.reindex(df_sorted.index).values

        return df_sorted


    def get_ppl(self, model, tokenizer, device, sample):
        """
        Calculate perplexity for a given text sample using a causal language model
        by sliding a window across the tokenized input text
        and averaging the negative log-likelihood across all windows.

        Args:
            model (torch.nn.Module): The causal language model to use for perplexity calculation
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the model
            device (torch.device): The device to run the model on (CPU or CUDA)
            sample (list of str): List of text segments to calculate perplexity on (will be joined with newlines)
        
        Returns:
            float: The perplexity score for the given text sample
        """


        #Get the enodings and set the stride
        encodings = tokenizer("\n\n".join(sample), return_tensors="pt")
        max_length = model.config.max_position_embeddings
        stride = 128
        seq_len = encodings.input_ids.size(1)

        #Get perplexity over the stride until everything is covered
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        #Average the score over sample length
        ppl = torch.exp(torch.stack(nlls).mean()).item()

        return ppl