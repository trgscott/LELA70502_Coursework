# LELA70502_Coursework

<h3 align="center">Can language models generate more original ideas?</h3>

  <p align="center">
    Natural language generation - directed reading assignment
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

This code relates to an assignment reporting on the ability of language models to generate original ideas.
The code tests whether fine-tuning GPT-2 with fictional content and adjusting the decoding paramaters of temperature and top-K can align a model to generate more original output.
This is tested via the semantic difference between generated texts using a self-METEOR score and ROUGE-L.
A qualitative test is made using brainteaser puzzles to test the originality and coherence of the text generated.


<!-- GETTING STARTED -->
## Getting Started

All of the code required to run the project is in the Can_language_models_generate_more_original_ideas?.ipynb file, which can be run in Colab.
This can be run on the CPU as the code does not take too long to run but it is set up to run on the GPU, so a GPU runtime will need to be selected.

### Prerequisites

These are the libraries that will need to be installed:
* Installations
  ```sh
  !pip install transformers -U
  !pip install evaluate
  !pip install rouge_score
  ```
  * Libraries
  ```sh
  import pandas as pd
  import re
  import gzip
  import json
  import numpy as np
  import torch
  import time
  import datetime
  from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, get_linear_schedule_with_warmup
  from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
  import random
  from tqdm import tqdm
  from random import randint
  import evaluate
  import nltk
  from nltk.translate import meteor
  nltk.download('punkt_tab')
  nltk.download('wordnet')
  from nltk import word_tokenize
  from nltk.corpus import WordNetCorpusReader, wordnet
  from sklearn.model_selection import train_test_split
  ```

<!-- USAGE EXAMPLES -->
## Usage

The code can be run entirely to produce the generations and test results for an example range of decoding parameters.
The decoding parameters in the assignment were varied from 1.0-1.5-3.0-5.0 for temperature and from 25-50-100 for the top-k value. These changes will need to be made in the code if you would like to see the differences in output - where to change these parameters is marked with a capitalised comment.
Here is an example answer to the brainteaser "Both persons who were playing chess won. What caused this to happen?" after fine-tuning on fictional content and setting the temperature to 3.0 and k to 50:

‘Could some mysterious person involved have prepared traps hidden through the manipulation game with instructions such as winning conditions and playing numbers incorrectly before them in advance of game administration?',

