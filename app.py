import streamlit as st
st.set_page_config(layout="wide")

st.title("Therapy Chatbots")
st.header("Three amazing artificial intelligences that you can talk to! Please wait for this webpage to fully load. It will take a minute or two.")

import tensorflow as tf
import pandas as pd
import cv2
import plotly.express as px
import numpy as np
import nltk
import matplotlib.pyplot as plt
import re
import spacy
import tensorflow_datasets as tfds
tf.random.set_seed(1)
from tensorflow.keras import layers , activations , models , preprocessing, utils
from keras import Input, Model
from keras.activations import softmax
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
tf.random.set_seed(1)
from google.colab import drive
import time
import os
import gdown


def tokenize(input):
  tokenizer = nltk.RegexpTokenizer(r"\w+")
  tokens = tokenizer.tokenize(input) ## YOUR CODE HERE
  return tokens

hello_keywords = ["hello", "hi", "hey"]
goodbye_keywords = ["bye", "goodbye","farewell"]

hello_response = "Hi!"
goodbye_response = "See you later!"

default_response = "Sorry, I'm trying, but I don't understand."

feel_keywords = ["feel", "felt", "feeling"]
emotions = ["happy", "sad", "scared", "content"]

def chatbot_2(input):
  # Convert input to lowercase
  userInput = input.lower()

  # Tokenize input and remove punctuation
  input_tokens = tokenize(userInput)
  # Respond to hello keyword if present
  for keyword in hello_keywords:
    if keyword in input_tokens:
      return hello_response
  # Respond to goodbye keyword if present
  for keyword in goodbye_keywords:
    if keyword in input_tokens:
      return goodbye_response
  for word in feel_keywords:
    if word in input_tokens:
      for emotion in emotions:
        if emotion in input_tokens:
          return "Why are you feeling " + emotion + "?"
  
  # Else respond with default response
  return default_response

st.title("ELIZA-like Chatbot")

prompt = st.text_input('This chatbot was created using simple rules. It responds to greetings, goodbyes, and simple feelings.', 'I feel sad.')
st.write('The chatbot\'s response: ', chatbot_2(prompt))

st.text("")
st.text("")
st.text("")



enc = tf.keras.models.load_model("Module2_Encoder")
dec = tf.keras.models.load_model("Module2_Decoder")

chat_data = pd.read_csv("https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv")

X = chat_data["questionText"] # TODO: replace None with the columns we should assign to X
y = chat_data["answerText"] # TODO: replace None with the columns we should assign to Y

def preprocess_text(phrase): 
  phrase = re.sub(r"\xa0", "", phrase) # YOUR CODE HERE: get rid of "\xa0"
  phrase = re.sub(r"\n", "", phrase) # YOUR CODE HERE: get rid of "\n"
  phrase = re.sub("[.]{1,}", ".", phrase) # removes duplicate "."s
  phrase = re.sub("[ ]{1,}", " ", phrase) # removes duplicate spaces
  return phrase

X = X.apply(preprocess_text)
y = y.apply(preprocess_text)


@st.cache
def getQAPairs(X, y):
  question_answer_pairs = []
  question_answer_pairs2 = []

  MAX_LENGTH = 100 # the maximum length for our sequences

  for (question, answer) in zip(X, y):
    question = preprocess_text(question) 
    answer = preprocess_text(answer)

    # split up question and answer into their constituent sentences

    question_arr = question.split(".") ## YOUR CODE HERE
    answer_arr = answer.split(".") ## YOUR CODE HERE

    # get the maximum number of question/answer pairs we can form,
    # which will be the shorter of len(question_arr) and len(answer_arr)

    max_sentences = min(len(question_arr), len(answer_arr)) ## YOUR CODE HERE

    for i in range(max_sentences):
      q_a_pair = []

      # get maximum sentence length
      max_q_length = min(MAX_LENGTH, len(question_arr[i]))
      max_a_length = min(MAX_LENGTH, len(answer_arr[i]))

      # append question, answer to pair (e.g,. first sentence of question + first sentence of answer, etc.)
      question_to_append = question_arr[i][0:max_q_length]
      q_a_pair.append(question_to_append)

      answer_to_append = "<START> " + answer_arr[i][0:max_a_length] + " <END>"
      q_a_pair.append(answer_to_append)

      question_answer_pairs.append(q_a_pair)

      # set up Q/A pair
      q_a_pair2 = []

      # append question, answer to pair (e.g,. first sentence of question + first sentence of answer, etc.)
      q_a_pair2.append(question_arr[i])
      q_a_pair2.append(answer_arr[i])

      # append to question_answer_pairs
      question_answer_pairs2.append(q_a_pair2)
  return question_answer_pairs, question_answer_pairs2

question_answer_pairs, question_answer_pairs2 = getQAPairs(X, y)

def tokenize(sentence):
  tokens = sentence.split(" ")
  return tokens

# re-create questions, answers
questions = [arr[0] for arr in question_answer_pairs]
answers = [arr[1] for arr in question_answer_pairs]

target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'
tokenizer = Tokenizer(filters=target_regex)
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1

# create encoder input data
tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max([len(x) for x in tokenized_questions])

# create decoder input data
tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max([len(x) for x in tokenized_answers])

def str_to_tokens(sentence: str):
    words = sentence.lower().split()
    tokens_list = list()
    for current_word in words:
        result = tokenizer.word_index.get(current_word, '')
        if result != '':
            tokens_list.append(result)
    return pad_sequences([tokens_list],
                        maxlen=maxlen_questions,
                        padding='post')
    
def LSTMChatbot(question):
  states_values = enc.predict(
      str_to_tokens(question))
  empty_target_seq = np.zeros((1, 1))
  empty_target_seq[0, 0] = tokenizer.word_index['start']
  stop_condition = False
  decoded_translation = ''
  while not stop_condition:
      dec_outputs, h, c = dec.predict([empty_target_seq]
                                            + states_values)
      sampled_word_index = np.argmax(dec_outputs[0, -1, :])
      sampled_word = None
      for word, index in tokenizer.word_index.items():
          if sampled_word_index == index:
              if word != 'end':
                  decoded_translation += ' {}'.format(word)
              sampled_word = word

      if sampled_word == 'end' \
              or len(decoded_translation.split()) \
              > maxlen_answers:
          stop_condition = True

      empty_target_seq = np.zeros((1, 1))
      empty_target_seq[0, 0] = sampled_word_index
      states_values = [h, c]
  return decoded_translation

st.title("Seq2Seq Model")

prompt2 = st.text_input('This chatbot was created using long-short term memory networks, which are a special type of recurrent neural networks.', 'I feel sad.')
st.write('The chatbot\'s response: ', LSTMChatbot(prompt2))

st.text("")
st.text("")
st.text("")



def scaled_dot_product_attention(query, key, value, mask):
  """Calculate the attention weights. """
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to zero out padding tokens
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output

class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # scaled dot-product attention
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs

def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, sequence length)
  return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)

class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

# complete encoder architecture
def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = encoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


# decoder itself
def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = decoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)
  # mask the future tokens for decoder inputs at the 1st attention block
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)
  # mask the encoder outputs for the 2nd attention block
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  enc_outputs = encoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    [arr[0] + arr[1] for arr in question_answer_pairs2], target_vocab_size=2**13)

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

# maximum sentence length
MAX_LENGTH = 100 # chosen arbitrarily

# tokenize, filter, pad sentences
def tokenize_and_filter(inputs, outputs):
  """
    Tokenize, filter, and pad our inputs and outputs
  """

  # store results
  tokenized_inputs, tokenized_outputs = [], []

  # loop through inputs, outputs
  for (sentence1, sentence2) in zip(inputs, outputs):

    # tokenize sentence
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
    
    # check tokenized sentence max length
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)

  # pad tokenized sentences
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen = MAX_LENGTH, padding = "post")
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen = MAX_LENGTH, padding = "post")
    
  return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter([arr[0] for arr in question_answer_pairs2], 
                                         [arr[1] for arr in question_answer_pairs2])

BATCH_SIZE = 64
BUFFER_SIZE = 20000

# decoder inputs use the previous target as input
# remove START_TOKEN from targets
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

tf.keras.backend.clear_session()
NUM_LAYERS = 2
UNITS = 512
D_MODEL = 256
NUM_HEADS = 8
DROPOUT = 0.1

model = transformer(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT)

def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
  # ensure labels have shape (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

model.load_weights("chatbot_transformer_v4.h5")

def evaluate(sentence):
  sentence = preprocess_text(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)

def predict(sentence):
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  return predicted_sentence

st.title("Transformer Model")

prompt3 = st.text_input('This chatbot was created using a transformer model, which uses attention to remember important words.', 'I feel sad.')
st.write('The chatbot\'s response: ', predict(prompt3))