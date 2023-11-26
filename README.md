# NeuralNetsNLP
20NewsGroup text classification models project for Queen's CMPE 452.

## Overview
This repository contains a series of Jupyter notebooks and Python scripts implementing different NLP models for text classification of the 20NewsGroup dataset. Each model explores different neural network architectures and embeddings to classify text data.

## Dataset
The dataset can be viewed and downloaded from [this link](http://qwone.com/~jason/20Newsgroups/).

## Models
The models included in this repository are:
- `GloveCNNModel.ipynb`: A Convolutional Neural Network (CNN) model using GloVe embeddings.
- `GloveLSTMModel.ipynb`: A Long Short-Term Memory (LSTM) model using GloVe embeddings.
- `GloveLSTMwithCNNModel.ipynb`: A model combining LSTM and CNN using GloVe embeddings.
- `word2vec_CNN.ipynb`: A CNN model using Word2Vec embeddings.
- `word2vec_CNN_pretrained.ipynb`: A CNN model with pre-trained Word2Vec embeddings.
- `word2vec_CNN_with_LSTM_pretrained.ipynb`: A model combining LSTM and CNN with pre-trained Word2Vec embeddings.
- `word2vec_LSTM.ipynb`: An LSTM model using Word2Vec embeddings.
- `word2vec_LSTM_pretrained.ipynb`: An LSTM model with pre-trained Word2Vec embeddings.

In addition to these models, the repository includes the following:
- `preprocessing.py`: A Python script for text data preprocessing which is used across all models.
- `ModelComparison.ipynb`: A notebook for comparing the performance of the different models.

## Contributions
Contributors to this project include:
- Jessica Guetre - [jessica-guetre](https://github.com/jessica-guetre)
- Devin Garrow - [devgarrow](https://github.com/devgarrow)
- Tom Hamilton - [t-hamilton20](https://github.com/t-hamilton20)
- Andy Craig

## Usage
To run the notebooks, Jupyter Notebook or JupyterLab are required.
```bash
git clone https://github.com/devgarrow/NeuralNetsNLP.git
cd NeuralNetsNLP
pip install -r requirements.txt
jupyter notebook
