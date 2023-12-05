# NLP Demo

## [Nlp From Scratch: Classifying Names With A Character-Level Rnn](./CharacterLevelDemo.py)

* Script and Dataset:
  > data/CharacterLevel.py \
  > model/rnn.py \
  > CharacterLevelDemo.py \
  > dataset/character
* Output 
  > Output/CharacterLevelDemo_xxx.png
* Reference: [PyTorch Tutorials](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
  
## [NLP: Text Classification](./AGnewsDemo.py)

* Script and Dataset:
  > data/agnews.py \
  > model/rnn.py \
  > model/lstm.py \
  > AGnewsDemo.py
* Reference: [AI for Beginners](https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/5-NLP/16-RNN/RNNPyTorch.ipynb)
  
## [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](./CharacterTranslationDemo.py)

* Script and Dataset:
  > data/CharacterTranslation.py \
  > model/rnn.py \
  > model/attentionrnn.py \
  > CharacterTranslationDemo.py \
  > dataset/data
* Output 
  > Output/CharacterTranslationDemo_xxx.png
* Reference: [PyTorch Tutorials](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

## Quick Start

### 1. Install environment
```
conda env create -f environment.yml
```

### 2. Download the dataset

See the [Dataset Download](dataset/download.md), the data will be saved in `dataset` directory.

### 3. Run the `xxxdemo.py` script
