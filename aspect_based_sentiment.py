# Setting  environment variables
import os
from urllib.request import urlretrieve
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  


# List of libraries to check
libraries = ['numpy', 'pandas', 'nltk', 'tensorflow', 'keras', 'pywsd', 'wget']


# Checking if libraries are installed and installing if they're not
for lib in libraries:
    try:
        __import__(lib)
    except ImportError:
        print(f"\n{lib} is not installed. Installing now...")
        os.system(f'pip install {lib}')

# Check if NLTK data resources are installed
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/averaged_perceptron_tagger')
    nltk.data.find('corpora/omw-1.4')
except (LookupError, FileNotFoundError):
    print("\nDownloading NLTK data resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')


# print("\n")
# print("""# This code expects the IMDb movie review database to be in the same directory as the .py file.
# # If the dataset is not availbale in "aclImdb" directory, this code will download it from:
# # https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# """)
# print("\n")



import numpy as np
from nltk.corpus import wordnet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, GRU, SimpleRNN, Dropout, BatchNormalization, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
import re
import string
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
import os, pathlib, shutil, random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
import wget

# def set_seeds(seed=42):
#     np.random.seed(seed)
#     random.seed(seed)
#     tf.random.set_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)

# set_seeds()


base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"

# Check if the dataset is already exists, if not, then download 
if not base_dir.exists():
  print(f"Dataset not exists in {base_dir}")
  script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
  # Download the dataset
  dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  download_filepath = os.path.join(script_dir, "aclImdb_v1.tar.gz")
  try:
    print(f"Downloading the dataset, plesse wait........")
    wget.download(dataset_url, download_filepath)
  except Exception as e:
    raise OSError(f"Error downloading dataset: {e}")

  # Unzip the dataset
  try:
    import tarfile  # Import tarfile after download to avoid unnecessary import
    with tarfile.open(download_filepath, "r:gz") as tar_file:
      print(f"\nUnzipping the dataset, plesse wait........")
      tar_file.extractall(script_dir)
  except Exception as e:
    raise OSError(f"Error unzipping dataset: {e}")

  print(f"Dataset downloaded and extracted to: {script_dir}")




# Check if val folder already exists, if so, delete it
if val_dir.exists():
    print("Deleting existing val folder...")
    shutil.rmtree(val_dir)


unsup_dir = base_dir / "train/unsup"
# Check if unsup folder exists, if so, delete it
if unsup_dir.exists():
    print("Deleting existing unsup folder...")
    shutil.rmtree(unsup_dir)


# Create new val folder
for category in ("neg", "pos"):
    os.makedirs(val_dir / category)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)
    num_val_samples = int(0.2 * len(files))
    val_files = files[-num_val_samples:]
    for fname in val_files:
        shutil.move(train_dir / category / fname, val_dir / category / fname)

print("Validation set preparation completed.")



# Load dataset
batch_size = 32
train_ds = keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory("aclImdb/val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=batch_size)





# Experiment1: Performs text classification using TextVectorization for unigrams, bigrams, and TF-IDF bigrams.
def experiment1():
  
  print("\n") 
  print("Experiment 1 Started\n")
  print("Preprocessing with TextVectorization (One-hot encoding for unigram and bigram)\n")
  # Preprocessing with TextVectorization
  text_vectorization = TextVectorization(
      max_tokens=20000,  # Limit vocabulary size
      output_mode="multi_hot"  # One-hot encoding for unigrams and bigrams
  )
  text_only_train_ds = train_ds.map(lambda x, y: x)  # Extract text from dataset
  text_vectorization.adapt(text_only_train_ds)  # Learn vocabulary from training data

  # Unigram model (single words)
  unigram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
  unigram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y))
  unigram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y))
  print("Builing model for unigram\n")
  model = get_modelForExp1()
  model.summary()
  callbacks = [
      keras.callbacks.ModelCheckpoint("unigram.keras", save_best_only=True)
  ]

  
  model.fit(unigram_train_ds.cache(),
            validation_data=unigram_val_ds.cache(),
            epochs=10,
            callbacks=callbacks)
  print("Loading the model for unigram\n")
  model = keras.models.load_model("unigram.keras")
  print(f"Result of Experiment 1 using Unigram Model")
  print(f"Test acc: {model.evaluate(unigram_test_ds)[1]:.3f}")



  # Bigram model (word pairs)
  text_vectorization = TextVectorization(
      max_tokens=20000,
      output_mode="multi_hot",
      ngrams=2  # Consider word pairs
  )
  text_vectorization.adapt(text_only_train_ds)
  bigram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y))
  bigram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y))
  bigram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y))
  print("\n\nBuiling model for Bigram\n")
  model = get_modelForExp1()
  model.summary()
  callbacks = [
      keras.callbacks.ModelCheckpoint("bigram.keras", save_best_only=True)
  ]
  model.fit(bigram_train_ds.cache(),
            validation_data=bigram_val_ds.cache(),
            epochs=10,
            callbacks=callbacks)
  print("Loading the model for Bigram\n")
  model = keras.models.load_model("bigram.keras")
  print(f"Result of Experiment 1 using Bigram Model")
  print(f"Test acc: {model.evaluate(bigram_test_ds)[1]:.3f}")
  

  
  # model-building utility function for Exp1
def get_modelForExp1(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"])
    return model


#execute experiment1
experiment1()

print("\n\nAspect Based Sentiment Analysis (ABSA) has started.... \n")
#:::::::::::::::: Code Starts for Specific Experiment2 ::::::::::::::::::::::::
# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Define base aspect list
aspect_list = [
    'movie','plot', 'storyline', 'screenplay', 'twist', 'ending', 'resolution', 'cast', 'interval', 'characters'
]

# Tokenize and pad the reviews
tokenizerExp2 = Tokenizer(num_words=20000, oov_token='<OOV>')


# Function to get synonyms using WordNet
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

# Tokenize the text data
texts = []
for text, _ in train_ds:
    decoded_text = text[0].numpy().decode('utf-8')
    texts.append(decoded_text)


# Fit the tokenizer on the text data
tokenizerExp2.fit_on_texts(texts)


max_len = 100  # Define maximum sequence length

def tokenize_and_pad(texts, tokenizer):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences


def generate_aspects(input_text, predefined_aspects):
    words = input_text.split()
    generated_aspects = []
    for word in words:
        sense = lesk(input_text, word)
        if sense:
            lemma = lemmatizer.lemmatize(sense.name().split('.')[0].lower())
            if lemma in predefined_aspects:
                generated_aspects.append(lemma)
    return generated_aspects


def preprocess_dataset(dataset, predefined_aspects):
    texts = []
    generated_aspects = []
    for text, _ in dataset:
        decoded_text = text[0].numpy().decode('utf-8')
        aspects = generate_aspects(decoded_text, predefined_aspects)
        texts.append(decoded_text)
        generated_aspects.append(aspects)
    return texts, generated_aspects

print("\n")
print("Preprocessing Dataset for Experiement 2 (ABSA using Bi-GRU Model)")
train_texts2, train_aspects2 = preprocess_dataset(train_ds, aspect_list)
val_texts2, val_aspects2 = preprocess_dataset(val_ds, aspect_list)
test_texts2, test_aspects2 = preprocess_dataset(test_ds, aspect_list)

X_train_exp2 = tokenize_and_pad(train_texts2, tokenizerExp2)
X_val_exp2 = tokenize_and_pad(val_texts2, tokenizerExp2)
X_test_exp2 = tokenize_and_pad(test_texts2, tokenizerExp2)

# Convert aspect data to one-hot encoding
def encode_aspect(aspects):
    encoding = []
    for aspect in aspects:
        aspect_encoding = [0] * len(aspect_list)
        for word in aspect:
            if word in aspect_list:
                aspect_encoding[aspect_list.index(word)] = 1
        encoding.append(aspect_encoding)
    return encoding

y_train2 = np.array(encode_aspect(train_aspects2))
y_val2 = np.array(encode_aspect(val_aspects2))
y_test2 = np.array(encode_aspect(test_aspects2))


# Define the ABSA model with GRU
model = keras.Sequential([
    Embedding(input_dim=20000, output_dim=100, input_length=max_len),
    SpatialDropout1D(0.2),
    Bidirectional(GRU(64, return_sequences=True)),
    Dropout(0.2),
    BatchNormalization(),
    Bidirectional(GRU(32, return_sequences=True)),
    Dropout(0.2),
    BatchNormalization(),
    GRU(16, return_sequences=False),
    Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    Dense(len(aspect_list), activation='softmax')
])



print("\n")
print("Compiling the Bi-GRU model")
# Define Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n")
print("Traning the model for Experiment 2")
# Train the model
history_exp2 = model.fit(
    X_train_exp2, y_train2,
    epochs=10,
    batch_size=32,
    validation_data=(X_val_exp2, y_val2),
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_exp2, y_test2)
print(f"Result of Experiment 2 using Bi-GRU Model")
print(f"Test accuracy: {test_acc:.3f}")
print(f"Test loss : {test_loss:.3f}")

#:::::::::::::::: Code Ends for Experiment2 ::::::::::::::::::::::::








#:::::::::::::::::::::::: Code Starts for Experiment3:::::::::::::::::
tokenizerExp3 = Tokenizer(num_words=20000, oov_token='<OOV>')

# Text preprocessing functions
def remove_html_tags(text):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',text)
    return text

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)

def remove_punctuation(text):
    return text.translate(str.maketrans('','',string.punctuation))

stop_words = stopwords.words('english')

def stopwords_removal(text):
    new_text=[]
    for word in text.split():
        if word in stop_words:
            new_text.append('')
        else:
            new_text.append(word)
    x=new_text[:]
    new_text.clear()
    return " ".join(x)  

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# Tokenize the text data
texts_exp3 = []
for text, _ in train_ds:
    decoded_text = text[0].numpy().decode('utf-8')
    # Apply preprocessing steps
    processed_text = remove_html_tags(decoded_text)
    processed_text = remove_url(processed_text)
    processed_text = remove_punctuation(processed_text)
    processed_text = stopwords_removal(processed_text)
    processed_text = remove_emoji(processed_text)
    texts_exp3.append(processed_text)

# Fit the tokenizer on the text data
tokenizerExp3.fit_on_texts(texts_exp3)


print("\n")
print("Preprocessing Dataset for Experiement 3 (ABSA using improved Bi-GRU Model)")
train_texts3, train_aspects3 = preprocess_dataset(train_ds, aspect_list)
val_texts3, val_aspects3 = preprocess_dataset(val_ds, aspect_list)
test_texts3, test_aspects3 = preprocess_dataset(test_ds, aspect_list)


X_train_exp3 = tokenize_and_pad(train_texts3, tokenizerExp3)
X_val_exp3 = tokenize_and_pad(val_texts3, tokenizerExp3)
X_test_exp3 = tokenize_and_pad(test_texts3, tokenizerExp3)

y_train3 = np.array(encode_aspect(train_aspects3))
y_val3 = np.array(encode_aspect(val_aspects3))
y_test3 = np.array(encode_aspect(test_aspects3))


print("Traning the model for Experiment 3")
# Train the model
history_exp3 = model.fit(
    X_train_exp3, y_train3,
    epochs=10,
    batch_size=32,
    validation_data=(X_val_exp3, y_val3),
    callbacks=[early_stopping]
)


# Evaluate the model
test_loss_exp3, test_acc_exp3 = model.evaluate(X_test_exp3, y_test3)
print(f"Result of Experiment 3 using improved Bi-GRU Model")
print(f"Test accuracy: {test_acc_exp3:.3f}")
print(f"Test loss : {test_loss_exp3:.3f}")
#:::::::::::::::: Code Ends for Experiment3 ::::::::::::::::::::::::


