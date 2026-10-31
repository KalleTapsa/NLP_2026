# NLP_2026
Analysis of stopwords and intent prediction in the MultiWOZ 2.2 dataset.

All code is in **`project7b_clean_with_outputs.ipynb`**. The notebook already contains the results of a full run (tables, printed results and plots), so you can read it directly on GitHub without running anything. The explanations and conclusions are in the project report.

The notebook is divided into ten parts:

| Part | Topic |
|---|---|
| 1 | Stopwords per topic, χ² test and residuals |
| 2 | How well the NLTK stopword list fits each topic |
| 3 | Named entities and the stopwords next to them |
| 4 | Vocabulary growth and Heaps' law |
| 5 | Semantic content of the topics with Empath |
| 6 | Predicting the intent with Empath |
| 7 | Predicting the intent with word and sentence embeddings |
| 8 | A trained classifier (TF-IDF + logistic regression) |
| 9 | Predicting the intent with an LLM (Phi-3.5-mini-instruct), compared with ConvLab's T5 |
| 10 | A more reliable evaluation: confidence intervals, context split and prompt robustness |

## Running the notebook yourself

### Requirements

- **Python 3.12** (tested). spaCy does not yet install on the newest Python versions, so use 3.12 or 3.13.
- About **15 GB of free disk space**, mostly for the downloaded models.
- At least **16 GB of RAM**. Phi-3.5-mini needs about 8 GB on its own.
- A GPU is optional. The notebook uses an NVIDIA GPU (CUDA) or an Apple Silicon GPU (MPS) if one is available, and the CPU otherwise.
- An internet connection for the first run, to download the data and the models.

### Setup

1. Create and activate a virtual environment in the project folder:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install jupyter ipykernel
   ```

2. Open `project7b_clean_with_outputs.ipynb` in Jupyter or VS Code and select the `.venv` environment as the kernel.

3. Run the first code cell. It installs all the needed packages:

   ```
   nltk numpy pandas matplotlib scipy certifi jinja2 spacy wordcloud empath
   scikit-learn gensim torch transformers sentence-transformers accelerate "preshed<3.0.14"
   ```

   `preshed` is limited to below 3.0.14 because that release breaks spaCy. Restart the kernel after the installation.

### Running

Run all cells from top to bottom. The notebook downloads everything it needs by itself:

| What | Where it goes | Size |
|---|---|---|
| MultiWOZ 2.2 data from the [official repository](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2) | `data/MultiWOZ_2.2/` | about 280 MB |
| NLTK stopword list | NLTK's own data folder | small |
| spaCy model `en_core_web_sm` | the Python environment | about 15 MB |
| RoBERTa-base, Sentence-Transformer `all-MiniLM-L6-v2` | Hugging Face cache | about 600 MB |
| `microsoft/Phi-3.5-mini-instruct` | Hugging Face cache | about 7.6 GB |
| `ConvLab/t5-small-nlu-multiwoz21-context3` | Hugging Face cache | about 250 MB |

The data is only downloaded once. Later runs use the files in `data/`. All plots are also saved as PNG files in the `figures/` folder. Running the notebook replaces the saved outputs with new ones.

### Running time

A full run takes several hours on a laptop. The slowest parts are:

- **Part 3:** spaCy reads all 143,000 utterances.
- **Part 7:** RoBERTa and the Sentence-Transformer embed all user utterances.
- **Parts 9 and 10:** Phi-3.5 classifies 1,000 utterances five times, once without context, once with context and three times with other prompt examples.

Parts 1–6 and Part 8 run in a few minutes. If you only need the corpus analysis, you can stop after Part 5.

### Reproducibility

All random choices use fixed seeds, so the results are the same on every run:

- the 80/20 train/test split in Part 8 (`random_state=42`);
- the 1,000-utterance test sample in Part 9 (`random_state=42`);
- the prompt examples in Part 9 (seed 0) and Part 10 (seeds 1–3);
- the bootstrap in Part 10 (seed 0);
- word2vec and FastText in Part 7 (`seed=42`, one worker, and a fixed hash function instead of Python's random one).

Phi-3.5 and the other neural models can give very slightly different numbers on different hardware (GPU vs CPU), but the conclusions stay the same.

### Troubleshooting

- **SSL errors when downloading the data:** the notebook already uses `certifi` for this. If the error remains, run `pip install --upgrade certifi`.
- **spaCy fails to import or install:** check that you are on Python 3.12 or 3.13 and that `preshed` is below 3.0.14.
- **Out of memory in Part 9:** close other programs, or run on a machine with more memory, for example on [CSC](https://www.csc.fi/).
