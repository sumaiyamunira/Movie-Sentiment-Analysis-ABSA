# Sentiment Analysis with Aspect-Based Sentiment Analysis (ABSA)

## Introduction
This project performs sentiment analysis on movie reviews using different models and techniques. It includes experiments in text classification with unigrams, bigrams, and aspect-based sentiment analysis using a Bi-GRU model.

## Dataset
The dataset used is the IMDB movie review dataset from Stanford, which consists of labeled sentiment data for movie reviews.

## Setup
### Environment Setup
Ensure you have Python 3.x installed along with the necessary libraries listed in `requirements.txt`. You can install them using pip:
  pip install -r requirements.txt

### Dataset Setup
The dataset will be downloaded automatically upon running the scripts if it's not already available locally. The dataset includes training, validation, and test sets.

# Experiments

## Experiment 1: Text Classification

- **Unigram Model:** Uses single words for classification.
- **Bigram Model:** Uses pairs of consecutive words for classification.

## Experiment 2: Aspect-Based Sentiment Analysis (ABSA)

- **Bi-GRU Model:** Implements aspect-based sentiment analysis using a Bidirectional GRU neural network.

## Experiment 3: Improved ABSA with Preprocessing

- **Text Cleaning:** Removing HTML tags, URLs, punctuation, and emojis.
- **Tokenization:** Splitting text into meaningful tokens.
- **Stopword Removal:** Eliminating common words.
- **WordNet and Lesk Algorithm:** Utilizing lexical analysis for word sense disambiguation.
- **Embedding Layer:** Transforming words into vectors for semantic understanding.
- **Bidirectional GRU (Bi-GRU):** Processing sequences bidirectionally for improved context comprehension.
- **Dense Layers and Early Stopping:** Enhancing model training and evaluation for accurate sentiment analysis.

## Results

The results of each experiment are saved in the respective model files:

- **Unigram Model**: `unigram.keras`
- **Bigram Model**: `bigram.keras`
- **Aspect-Based Sentiment Analysis (ABSA)**: `aspect_based_sentiment_analysis.keras`

## Paper

You can read my paper related to this project on [LinkedIn](https://www.linkedin.com/in/sumaiya-munira/overlay/education/875692166/multiple-media-viewer/?profileId=ACoAABAQQNABg0DIA29wAiFm5JKFcVHAl-DLMB0&treasuryMediaId=1721247253277).

## Conclusion

This project demonstrates various techniques for sentiment analysis on movie reviews and provides insights into different model performances. The experiments highlight the effectiveness of different preprocessing techniques and neural network architectures in enhancing sentiment analysis accuracy.

## Authors

- Sumaiya Salam Munira

## References

- [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [NLTK Documentation](https://www.nltk.org/)






