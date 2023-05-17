# datasci-gpt

A HuggingFace tutorial

New training based off of gpt-2

Dataset is filtered to only contain data science terminology, to act as a smaller, quick code completion model for data science users

filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

The filtered dataset contains about 3% of the original dataset, resulting in a size of 6 GB consisting of 600,000 Python scripts
```
"We now have 16.7 million examples with 128 tokens each, which corresponds to about 2.1 billion tokens in total. 
For reference, OpenAI’s GPT-3 and Codex models are trained on 300 and 100 billion tokens, respectively, where the 
Codex models are initialized from the GPT-3 checkpoints. Our goal in this section is not to compete with these 
models, which can generate long, coherent texts, but to create a scaled-down version providing a quick autocomplete 
function for data scientists."
```

-----
files

load.ipynb - notebook containing the steps to take for loading the dataset, tokenizing, batching, and training

load.py - the python script derived from the code written in the load notebook
