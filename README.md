# Sentiment Analysis with Aspect-Based Sentiment Analysis (ABSA)

## Introduction
This project performs sentiment analysis on movie reviews using different models and techniques. It includes experiments in text classification with unigrams, bigrams, and aspect-based sentiment analysis using a Bi-GRU model.

## Dataset
The dataset used is the IMDB movie review dataset from Stanford, which consists of labeled sentiment data for movie reviews.

## Setup
### Environment Setup
Ensure you have Python 3.x installed along with the necessary libraries listed in `requirements.txt`. You can install them using pip:
  ```bash
  pip install -r requirements.txt

### Dataset Setup
The dataset will be downloaded automatically upon running the scripts if it's not already available locally. The dataset includes training, validation, and test sets.

# Experiments

## Experiment 1: Text Classification

- **Unigram Model:** Uses single words for classification.
- **Bigram Model:** Uses pairs of consecutive words for classification.

## Experiment 2: Aspect-Based Sentiment Analysis (ABSA)

- **Bi-GRU Model:** Implements aspect-based sentiment analysis using a Bidirectional GRU neural network.



