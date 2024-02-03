import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Sample movie reviews
positive_review = "This movie is fantastic! I loved every moment ."
negative_review = "I couldn't stand this movie. It was terrible and boring."

# Combine reviews into a list
reviews = [positive_review, negative_review]

# Labels (1 for positive, 0 for negative)
labels = [1, 0]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)

# Padding sequences to make them of equal length
max_sequence_length = max(len(sequence) for sequence in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert labels to categorical
categorical_labels = to_categorical(labels)

# Example of the processed data
# print(reviews)
# print("Max sequence length:")
# print(max_sequence_length)
# print("\nTokenized Sequences:")
# print(sequences)
# print("\nPadded Sequences:")
# print(padded_sequences)
# print("\nCategorical Labels:")
# print(categorical_labels)

# Define LSTM model
embedding_dim = 50  # Dimension of word embeddings
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size (+1 for padding token)
output_classes = 2  # Number of output classes (positive and negative)
max_sequence_length = padded_sequences.shape[1]  # Maximum sequence length
# print(tokenizer.word_index)### print the unique words
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(units=128))  # LSTM layer with 128 units
model.add(Dense(output_classes, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(padded_sequences, categorical_labels, epochs=10, batch_size=32)

# Define new sentences
new_sentences = ["This movie is amazing!", "I hated every moment of this film."]

# Tokenize and pad the new sentences
new_sequences = tokenizer.texts_to_sequences(new_sentences)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)

# Predict sentiment for new sentences
predictions = model.predict(new_padded_sequences)

# Convert predictions to labels
predicted_labels = np.argmax(predictions, axis=1)

# Map labels to sentiment
sentiment_map = {0: "Negative", 1: "Positive"}
predicted_sentiments = [sentiment_map[label] for label in predicted_labels]

# Print predictions
for sentence, sentiment in zip(new_sentences, predicted_sentiments):
    print(f"Sentence: '{sentence}' --> Predicted Sentiment: {sentiment}")
