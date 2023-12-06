# Concept in NLP


## 1. Concept

### One-hot vector

### Word Embedding

### Perplexity

### Temperature


## 2. Model Card

### RNN: Recurrent Neural Networks

### Bi-directional RNNs

### GRU: Gated Recurrent Units

### LSTM: Long Short-Term Memory

### Attention
* **Context Vector** is an expected value

### Transformer
* Transformer Model
  > **T5, GPT-2, BERT**
 
### Reformer: The Reversible Transformer


## 3. Metric

### F1 score

### BLEU: Bilingual Evaluation Understudy
* The closer to 1, the better

### ROUGE: Recall-Oriented Understudy for Gisting Evaluation

## GLUE: General Language Understanding Evaluation


## 4. Sampling and Decoding

### Random sampling

### Temperature in sampling

### Greedy Decoding
* **Lower temperature setting:** More confident, conservative network
* **Higher temperature setting:** More excited, random network

### Beam Search
* Problem:
  >1. Penalizes long sequences, so you should normalize by the sentence length
  >2. Computationally expensive and consumes a lot of memory

### MBR: Minimum Bayes Risk
