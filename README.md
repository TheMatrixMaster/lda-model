# Latent Dirichlet Allocation (LDA)

Simple reusable implementation of LDA in Python.

## Usage
You'll need to preprocess your documents into a list of lists of tokens `[[str]]`. Then, simply pass this data to the LDA class and choose the number of topics you want to model. See `run.py` for a full example.

## Results
Here are some results from running LDA on a truncated version of the mimic3 dataset. The dataset contains ~683 documents (patients), and ~1000 unique tokens (ICD9 codes). The model was run for 200 iterations with 5 topics.

I plotted the top 10 words for each topic, and the top 100 documents for each topic. I also plotted the correlation between the topics and some selected tokens.

![Alt text](/results/top_words.png?raw=true)
![Alt text](/results/word_topic_corr.png?raw=true)
![Alt text](/results/top_docs.png?raw=true)
