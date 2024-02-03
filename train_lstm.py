import os
import time
import data
from datetime import datetime
from deep_lstm import DLSTM
from utils import *

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "14000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "100"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "100"))
NEPOCH = int(os.environ.get("NEPOCH", "10"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "1000"))

if not MODEL_OUTPUT_FILE:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    MODEL_OUTPUT_FILE = "Forward_LSTM-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

x_train, y_train, word_to_index, index_to_word = data.get_train_sentences(VOCABULARY_SIZE)

model = DLSTM(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)

t1 = time.time()
loss = model.train_step(x_train, y_train)
t2 = time.time()
print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))
sys.stdout.flush()

def sgd_callback(model, num_examples_seen):
    dt = datetime.now().isoformat()
    loss = model.calculate_loss(x_train[:10000], y_train[:10000])
    print("\n%s (%d)" % (dt, num_examples_seen))
    print("--------------------------------------------------")
    print("Loss: %f" % loss)
    save_model_parameters_tf(model, MODEL_OUTPUT_FILE)
    print("\n")
    sys.stdout.flush()

for epoch in range(NEPOCH):
    train_with_sgd_tf(model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9,
                      callback_every=PRINT_EVERY, callback=sgd_callback)
