import numpy as np
from nltk import wordpunct_tokenize
import operator
import re, string
import math
from keras.preprocessing.sequence import pad_sequences
# /Users/mhnguyetvu/workspace/paraphrase-identification/data/MSR Paraphrase Test.txt
## Define token
SENTENCE_START_TOKEN = "sentence_start"
SENTENCE_END_TOKEN = "sentence_end"
UNKNOWN_TOKEN = "unknown_token"
PAD_TOKEN = "PAD"
def load_data(loc='/Users/mhnguyetvu/workspace/paraphrase-identification/data/', _train=False, _test=False):
    "Load the MSRP dataset."
    trainloc = loc + 'msr_paraphrase_train.txt'
    testloc = loc + 'msr_paraphrase_test.txt'

    sent1_train, sent2_train, sent1_test, sent2_test = [], [], [], []
    label_train, label_dev, label_test = [], [], []

    if _train:
        with open(trainloc, 'r', encoding='utf8') as f:
            f.readline()  # skipping the header of the file
            for line in f:
                text = line.strip().split('\t')
                sent1_train.append("%s %s %s" % (SENTENCE_START_TOKEN, text[3], SENTENCE_END_TOKEN))
                sent2_train.append("%s %s %s" % (SENTENCE_START_TOKEN, text[4], SENTENCE_END_TOKEN))
                label_train.append(int(text[0]))

    if _test:
        with open(testloc, 'r', encoding='utf8') as f:
            f.readline()  # skipping the header of the file
            for line in f:
                text = line.strip().split('\t')
                sent1_test.append("%s %s %s" % (SENTENCE_START_TOKEN, text[3], SENTENCE_END_TOKEN))
                sent2_test.append("%s %s %s" % (SENTENCE_START_TOKEN, text[4], SENTENCE_END_TOKEN))
                label_test.append(int(text[0]))

    if _train and _test:
        return [sent1_train, sent2_train], [sent1_test, sent2_test], [label_train, label_test]
    elif _train:
        return [sent1_train, sent2_train], label_train
    elif _test:
        return [sent1_test, sent2_test], label_test

# #########Example usage:
# # Load training and test data
# train_data, test_data, labels = load_data(_train=True, _test=True)

# # Access the training and test data
# train_sentences, test_sentences = train_data[0], test_data[0]

# # Access the labels
# train_labels, test_labels = labels[0], labels[1]

# # Print the first training sentence and its label
# print("First training sentence:", train_sentences[0])
# print("Corresponding label:", train_labels[0])

# # Print the first test sentence and its label
# print("First test sentence:", test_sentences[0])
# print("Corresponding label:", test_labels[0])

def my_tokenizer(input):
    """Tokenizer to tokenize and normalize text."""
    tokenList = []
    tokens = wordpunct_tokenize(input.lower())
    tokenList.extend([x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)])
    return tokenList
# # Example text
# input_text = "Hello! This is an example sentence. It contains punctuation, such as commas, periods, and exclamation marks!!!"

# # Tokenize and normalize the text
# tokens = my_tokenizer(input_text)

# # Print the tokens
# print("Tokens:", tokens)

# def build_dictionary(loc='/Users/mhnguyetvu/workspace/paraphrase-identification/data/', vocabulary_size=-1):
#     """Construct a dictionary from the MSRP dataset."""
#     trainloc = loc + 'msr_paraphrase_train.txt'
#     testloc = loc + 'msr_paraphrase_test.txt'
    
#     document_frequency = {}
#     total_document = 0
#     with open(trainloc, 'r', encoding='utf8') as f:
#         f.readline()  # skipping the header of the file
#         for line in f:
#             text = line.strip().split('\t')
#             sentence1 = my_tokenizer(text[3])
#             sentence2 = my_tokenizer(text[4])

#             for token in set(sentence1):
#                 if token in document_frequency:
#                     document_frequency[token] = document_frequency[token] + 1
#                 else:
#                     document_frequency[token] = 1

#             for token in set(sentence2):
#                 if token in document_frequency:
#                     document_frequency[token] = document_frequency[token] + 1
#                 else:
#                     document_frequency[token] = 1

#             total_document = total_document + 2

#     with open(testloc, 'r', encoding='utf8') as f:
#         f.readline()  # skipping the header of the file
#         for line in f:
#             text = line.strip().split('\t')
#             sentence1 = my_tokenizer(text[3])
#             sentence2 = my_tokenizer(text[4])

#             for token in set(sentence1):
#                 if token in document_frequency:
#                     document_frequency[token] = document_frequency[token] + 1
#                 else:
#                     document_frequency[token] = 1

#             for token in set(sentence2):
#                 if token in document_frequency:
#                     document_frequency[token] = document_frequency[token] + 1
#                 else:
#                     document_frequency[token] = 1

#             total_document = total_document + 2

#     for key, value in document_frequency.items():
#         document_frequency[key] = math.log(total_document / document_frequency[key])

#     vocab = sorted(document_frequency.items(), key=operator.itemgetter(1), reverse=True)

#     word_to_index = dict()
#     index_to_word = dict()
#     word_to_index[SENTENCE_START_TOKEN] = 0
#     word_to_index[SENTENCE_END_TOKEN] = 1
#     word_to_index[UNKNOWN_TOKEN] = 2
#     index_to_word[0] = SENTENCE_START_TOKEN
#     index_to_word[1] = SENTENCE_END_TOKEN
#     index_to_word[2] = UNKNOWN_TOKEN
#     # index_to_word[3] = PAD_TOKEN
    
#     counter = 3
#     for key, value in vocab:
#         if len(key) < 4:
#             continue
#         elif counter == vocabulary_size:
#             break
#         word_to_index[key] = counter
#         index_to_word[counter] = key
#         counter = counter + 1

#     return word_to_index, index_to_word
def build_dictionary(loc='./data/', vocabulary_size=-1):
    """Construct a dictionary from the MSRP dataset."""
    trainloc = loc + 'msr_paraphrase_train.txt'
    testloc = loc + 'msr_paraphrase_test.txt'

    document_frequency = {}
    total_document = 0
    with open(trainloc, 'r', encoding='utf8') as f:
        f.readline()  # skipping the header of the file
        for line in f:
            text = line.strip().split('\t')
            sentence1 = my_tokenizer(text[3])
            sentence2 = my_tokenizer(text[4])

            for token in set(sentence1):
                if token in document_frequency:
                    document_frequency[token] = document_frequency[token] + 1
                else:
                    document_frequency[token] = 1

            for token in set(sentence2):
                if token in document_frequency:
                    document_frequency[token] = document_frequency[token] + 1
                else:
                    document_frequency[token] = 1

            total_document = total_document + 2

    with open(testloc, 'r', encoding='utf8') as f:
        f.readline()  # skipping the header of the file
        for line in f:
            text = line.strip().split('\t')
            sentence1 = my_tokenizer(text[3])
            sentence2 = my_tokenizer(text[4])

            for token in set(sentence1):
                if token in document_frequency:
                    document_frequency[token] = document_frequency[token] + 1
                else:
                    document_frequency[token] = 1

            for token in set(sentence2):
                if token in document_frequency:
                    document_frequency[token] = document_frequency[token] + 1
                else:
                    document_frequency[token] = 1

            total_document = total_document + 2

    for key, value in document_frequency.items():
        document_frequency[key] = math.log(total_document / document_frequency[key])

    vocab = sorted(document_frequency.items(), key=operator.itemgetter(1), reverse=True)

    word_to_index = dict()
    index_to_word = dict()
    word_to_index[SENTENCE_START_TOKEN] = 0
    word_to_index[SENTENCE_END_TOKEN] = 1
    word_to_index[UNKNOWN_TOKEN] = 2
    word_to_index[PAD_TOKEN] = 3  # Add PAD_TOKEN here
    index_to_word[0] = SENTENCE_START_TOKEN
    index_to_word[1] = SENTENCE_END_TOKEN
    index_to_word[2] = UNKNOWN_TOKEN
    index_to_word[3] = PAD_TOKEN  # Add PAD_TOKEN here

    counter = 4
    for key, value in vocab:
        if len(key) < 4:
            continue
        elif counter == vocabulary_size:
            break
        word_to_index[key] = counter
        index_to_word[counter] = key
        counter = counter + 1

    return word_to_index, index_to_word
########################################### Example usage of build_dictionary
# vocabulary_size = 100  # Set your desired vocabulary size
# word_to_index, index_to_word = build_dictionary(loc = '/Users/mhnguyetvu/workspace/paraphrase-identification/data/', vocabulary_size=vocabulary_size)

# # Print some information for demonstration
# print(word_to_index)
# print("index_to_word:",index_to_word[10])
# print("Size of word_to_index dictionary:", len(word_to_index))
# print("Size of index_to_word dictionary:", len(index_to_word))
# print("Index of 'sentence_start' token:", word_to_index["sentence_start"])
# print("Word corresponding to index 10:", index_to_word[10])

def get_train_data(vocabulary_size, max_sentence_length=None):
    """Get training sentences with padding."""
    # Build dictionary
    word_to_index, index_to_word = build_dictionary(vocabulary_size=vocabulary_size)
    
    # Load training data
    [sent1_train, sent2_train], label_train = load_data(_train=True)

    # Tokenize and replace unknown tokens
    sent1_train_tokenized = [my_tokenizer(sent) for sent in sent1_train]
    sent2_train_tokenized = [my_tokenizer(sent) for sent in sent2_train]
    sent1_train_tokenized = [[w if w in word_to_index else UNKNOWN_TOKEN for w in sent] for sent in sent1_train_tokenized]
    sent2_train_tokenized = [[w if w in word_to_index else UNKNOWN_TOKEN for w in sent] for sent in sent2_train_tokenized]

    # Convert tokens to indices
    sent1_train_indices = [[word_to_index[word] for word in sentence] for sentence in sent1_train_tokenized]
    sent2_train_indices = [[word_to_index[word] for word in sentence] for sentence in sent2_train_tokenized]
    
    # Pad sequences if max_sentence_length is provided
    if max_sentence_length:
        sent1_train_indices = pad_sequences(sent1_train_indices, maxlen=max_sentence_length, padding='post', truncating='post')
        sent2_train_indices = pad_sequences(sent2_train_indices, maxlen=max_sentence_length, padding='post', truncating='post')

    return sent1_train_indices, sent2_train_indices, word_to_index, index_to_word, label_train
# Example usage of get_train_data function
# vocabulary_size = 100  # Set your desired vocabulary size
# X_train, y_train , word_to_index, index_to_word, label_train = get_train_data(vocabulary_size)

# # Print some information for demonstration
# print("Number of sentences in sent1_train:", len(y_train))
# print("Number of sentences in sent2_train:", len(sent2_train_indices))
# # print("Word 'I' has index:", word_to_index["I"])
# print("Index 10 corresponds to word:", index_to_word[10])
# print("Labels:", label_train)


def get_train_data_reversed(vocabulary_size):
    """Get training sentences in reversed order."""
    sent1_train_indices, sent2_train_indices, word_to_index, index_to_word, label_train = get_train_data(
        vocabulary_size)
    sent1_train_indices_reversed = []
    for index_list in sent1_train_indices:
        temp = []
        temp.extend(index_list)
        temp.reverse()
        sent1_train_indices_reversed.append(temp)

    sent2_train_indices_reversed = []
    for index_list in sent2_train_indices:
        temp = []
        temp.extend(index_list)
        temp.reverse()
        sent2_train_indices_reversed.append(temp)

    return sent1_train_indices_reversed, sent2_train_indices_reversed, word_to_index, index_to_word, label_train

def pad_sequence(sequence, max_length):
    """Pad sequence with zeros to match max_length."""
    padded_sequence = sequence + [0] * (max_length - len(sequence))
    return padded_sequence

# def get_train_sentences(vocabulary_size):
#     """Get training sentences with word to index map and vice versa."""
#     sent1_train_indices, sent2_train_indices, word_to_index, index_to_word, label_train = get_train_data(
#         vocabulary_size)
#     all_sentences = []
#     all_sentences.extend(sent1_train_indices)
#     all_sentences.extend(sent2_train_indices)
#     print(all_sentences)
#     X_train = np.asarray([[w for w in sentence[:-1]] for sentence in all_sentences])
#     y_train = np.asarray([[w for w in sentence[1:]] for sentence in all_sentences])

#     return X_train, y_train, word_to_index, index_to_word

def get_train_sentences(vocabulary_size):
    """Get training sentences with word to index map and vice versa."""
    sent1_train_indices, sent2_train_indices, word_to_index, index_to_word, label_train = get_train_data(
        vocabulary_size)
    
    # Combine indices of both sentences
    all_sentences = sent1_train_indices + sent2_train_indices

    # Find the maximum length among all sentences
    max_length = max(len(sentence) for sentence in all_sentences)

    # Pad or truncate sentences to ensure they all have the same length
    padded_sentences = [sentence + [word_to_index[PAD_TOKEN]] * (max_length - len(sentence)) for sentence in all_sentences]

    # Convert sentences to NumPy arrays
    X_train = np.array([sentence[:-1] for sentence in padded_sentences])
    y_train = np.array([sentence[1:] for sentence in padded_sentences])

    return X_train, y_train, word_to_index, index_to_word


def get_train_sentences_reversed(vocabulary_size):
    """Get training sentences in reverse order with word to index map and vice versa."""
    sent1_train_indices_reversed, sent2_train_indices_reversed, word_to_index, index_to_word, label_train = get_train_data_reversed(
        vocabulary_size)
    all_sentences = []
    all_sentences.extend(sent1_train_indices_reversed)
    all_sentences.extend(sent2_train_indices_reversed)

    X_train = np.asarray([[w for w in sentence[:-1]] for sentence in all_sentences])
    y_train = np.asarray([[w for w in sentence[1:]] for sentence in all_sentences])

    return X_train, y_train, word_to_index, index_to_word

def get_test_data(vocabulary_size):
    """Get testing sentences."""
    word_to_index, index_to_word = build_dictionary(vocabulary_size=vocabulary_size)
    [sent1_test, sent2_test], label_test = load_data(_test=True)

    sent1_test_tokenized = [my_tokenizer(sent) for sent in sent1_test]
    sent2_test_tokenized = [my_tokenizer(sent) for sent in sent2_test]

    for i, sent in enumerate(sent1_test_tokenized):
        sent1_test_tokenized[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]
    for i, sent in enumerate(sent2_test_tokenized):
        sent2_test_tokenized[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    sent1_test_indices = []
    for sentence in sent1_test_tokenized:
        sent1_test_indices.append([word_to_index[word] for word in sentence])

    sent2_test_indices = []
    for sentence in sent2_test_tokenized:
        sent2_test_indices.append([word_to_index[word] for word in sentence])

    return sent1_test_indices, sent2_test_indices, word_to_index, index_to_word, label_test

def get_test_data_reversed(vocabulary_size):
    """Get testing sentences in reverse order."""
    sent1_test_indices, sent2_test_indices, word_to_index, index_to_word, label_test = get_test_data(vocabulary_size)

    sent1_test_indices_reversed = []
    for index_list in sent1_test_indices:
        temp = []
        temp.extend(index_list)
        temp.reverse()
        sent1_test_indices_reversed.append(temp)

    sent2_test_indices_reversed = []
    for index_list in sent2_test_indices:
        temp = []
        temp.extend(index_list)
        temp.reverse()
        sent2_test_indices_reversed.append(temp)

    return sent1_test_indices_reversed, sent2_test_indices_reversed, word_to_index, index_to_word, label_test

def get_test_sentences(vocabulary_size):
    """Get testing sentences with word to index map and vice versa."""
    sent1_test_indices, sent2_test_indices, word_to_index, index_to_word, label_test = get_test_data(vocabulary_size)
    all_sentences = []
    all_sentences.extend(sent1_test_indices)
    all_sentences.extend(sent2_test_indices)

    x_test = np.asarray([[w for w in sentence] for sentence in all_sentences])

    return x_test, word_to_index, index_to_word


def get_test_sentences_reversed(vocabulary_size):
    """Get testing sentences in reverse order with word to index map and vice versa."""
    sent1_test_indices, sent2_test_indices, word_to_index, index_to_word, label_test = get_test_data_reversed(
        vocabulary_size)
    all_sentences = []
    all_sentences.extend(sent1_test_indices)
    all_sentences.extend(sent2_test_indices)

    x_test = np.asarray([[w for w in sentence] for sentence in all_sentences])

    return x_test, word_to_index, index_to_word

