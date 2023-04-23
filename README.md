# datasci-gpt

New training based off of gpt-2

Dataset is filtered to only contain data science terminology, to act as a smaller, quick code completion model for data science users

filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

The filtered dataset contains about 3% of the original dataset, resulting in a size of 6 GB consisting of 600,000 Python scripts

-----
files

load.ipynb - notebook containing the steps to take for loading the dataset, tokenizing, batching, and training

load.py - the python script derived from the code written in the load notebook
