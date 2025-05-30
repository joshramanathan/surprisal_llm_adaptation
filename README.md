# Adaptive Pre-Training of LLMs to Represent a Population's Comprehension Ability

This repository provides a package containing the full experimental pipeline for evaluating the effect of adaptive pre-training on language models to better represent the reading comprehension of specific populations.

More specifically, the goal is to determine whether the estimated surprisal from a population-adapted model is a better predictor of reading time than that from a baseline (non-adapted) model. Here, a *population* is defined as English L2 speakers with a specific L1 background: German, Italian, Mandarin, Portuguese, Russian, or Turkish. Reading times are drawn from participants in the MECO project ([Kuperman et al., 2022](https://doi.org/10.1017/S0272263121000954)). Adapted models are trained on a combination of learner data (a cleaned subcorpus of EFCAMDAT: [Shatz, 2020](https://doi.org/10.1075/ijlcr.20009.sha)) and geographically localized English corpora from participants’ countries of origin (CGLU: [Dunn, 2020](https://doi.org/10.1007/s10579-020-09489-2)). For more information, [this paper]() (link to be added) describes the methodology and evaluation used.

## Table of Contents
1. [Installation](#installation)
2. [Obtaining Corpora](#obtaining-corpora)
3. [Directory Structure](#directory-structure)
4. [Usage](#usage)
5. [License](#license)

---

<a name="installation"></a>

## Installation

This package is not available on PyPI. To install:

1. Clone this repository:
   ```
   git clone https://github.com/joshramanathan/surprisal_llm_adaptation.git
   cd surprisal_llm_adaptation
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

It is recommended to use a virtual environment or conda environment.

---

<a name="obtaining-corpora"></a>

## Obtaining Corpora

Some corpora used for model adaptation are not included in this repository due to licensing restrictions. You will need to obtain them separately:

- **EFCAMDAT**: Request access to the EF-Cambridge Open Language Database (EFCAMDAT) corpus [here](https://corpus.mml.cam.ac.uk/efcamdat/).
    - Once you have access to the corpus, navigate to `Cleaned_Subcorpus (Shatz, 2020)` and download `Final database (alternative prompts).xlsx`.
    - Place this file in the data directory in `data/original_corpora/efcamdat`.
- **CGLU**: The Corpus of Global Language Use (CGLU) v5.2 is available [here](https://publicdata.canterbury.ac.nz/Research/Geocorpus/CGLU_v5.2/).
    - The entire corpus is not necessary; you must only download the following files and place them into their respective folders in the `data/original_corpora/cglu` directory without changing the file names.
        - [German](https://publicdata.canterbury.ac.nz/Research/Geocorpus/CGLU_v5.2/europe_west/Germany/eng/europe_west.Germany.eng.clean.IN.gz)
        - [Italian](https://publicdata.canterbury.ac.nz/Research/Geocorpus/CGLU_v5.2/europe_west/Italy/eng/europe_west.Italy.eng.clean.IN.gz)
        - [Portuguese](https://publicdata.canterbury.ac.nz/Research/Geocorpus/CGLU_v5.2/america_brazil/Brazil/eng/america_brazil.Brazil.eng.clean.IN.gz)
        - [Mandarin](https://publicdata.canterbury.ac.nz/Research/Geocorpus/CGLU_v5.2/asia_east/Taiwan/eng/asia_east.Taiwan.eng.clean.IN.gz)
        - [Russian](https://publicdata.canterbury.ac.nz/Research/Geocorpus/CGLU_v5.2/europe_russia/Russia/eng/europe_russia.Russia.eng.1.IN.gz)
        - [Turkish](https://publicdata.canterbury.ac.nz/Research/Geocorpus/CGLU_v5.2/middle_east/Turkey/eng/middle_east.Turkey.eng.clean.original.gz)

- **MECO**: The Multilingual Eye-movement Corpus (L2 release 2.0) is available [here](https://osf.io/q9h43/).
    - Download the entire `release 2.0` folder and place it in `data/original_corpora/meco/`.

Additionally, if desired, the `models` directory created as an artifact from this experiment can be downloaded from the associated OSF page [here](https://osf.io/n6ebg/), and placed in `data`.

---

<a name="directory-structure"></a>

## Directory Structure

**`data/`**

Directory containing all corpora and generated data. This may be named anything, but is `data` by default.

- `original_corpora/` The directory in which downloaded corpora must be placed (as described above).
  - `cglu/`
    - `German/`
    - `Italian/`
    - `Mandarin/`
    - `Portuguese/`
    - `Russian/`
    - `Turkish/`
  - `efcamdat/`
  - `meco/`

**`surprisal_llm_adaptation/`**

The package directory.

- `__init__.py`
- `runner.py` Contains the `ExperimentRunner` class for orchestrating training and evaluation.
- `surprisal_calc.py` Contains the `SurprisalCalculator` class
- `utilities.py` Helper variables and functions.

---

<a name="usage"></a>

## Usage

Once the corpora are organized as described above, you can run the full training and evaluation pipeline as conducted in the original experiment:

```bash
python main.py
```

This will create all artifact files and directories entirely within `data`. If you wish to modify the data directory or other parameters, you can adjust the `run_experiment` arguments inside `main.py`.

Alternatively, you can use the package manually:
```python
import surprisal_llm_adaptation

runner = surprisal_llm_adaptation.ExperimentRunner(model_id="EleutherAI/pythia-160m-deduped")
```

You may replace `model_id` with any HuggingFace model identifier for an autoregressive language model (e.g., GPT-2, Pythia).
Optionally, you can also specify a different `data_dir` path, but the new `data_dir` must be organized as described in [Directory Structure](#directory-structure).

To begin adaptation and evaluation:

```python
runner.run_experiment()
```

Or call individual steps separately:

```python
runner.get_efcamdat_dfs()
runner.get_cglu_dfs()
runner.combine_efcamdat_cglu()
runner.adapt_models()
runner.evaluate_model_production()
runner.evaluate_model_representation()
runner.get_meco_dfs()
runner.get_regression_dfs()
runner.fit_regression_models()
runner.graph_perplexities()
runner.graph_dlls()
```

NOTE: `full_pipeline()` and `adapt_models()` may optionally be passed the following parameters:

- **num_proc** (`int`, _optional_) - The number of processes (CPUs) to use in parallel during processing. Defaults to 24.
- **batch_size** (`int`, _optional_) - The number of samples to be processed together during tokenization. Defaults to 100.
- **per_device_train_batch_size** (`int`, _optional_) - The batch size to use per GPU. Defaults to 16.
- **block_size** (`int`, _optional_) - The number of tokens in a single block to be given to the model. Defaults to 2048.

It is recommended to adjust `num_proc` and `batch_size` according to your machine's capabilities. However, recreating the exact experiment necessarily requires the default parameters.

---

<a name="license"></a>

## License

This project is licensed under the GNU General Public License. See the [LICENSE](LICENSE) file for details.