import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
data = pd.read_csv("../poetry_data.csv")
data.dropna(inplace=True)

# Use only Hindi and English columns, drop S.NO
data = data[['Hindi Poetry', 'English Poetry']]
data.columns = ['input_text', 'target_text']
data['target_text'] = data['target_text'].apply(lambda x: '<start> ' + x + ' <end>')

# Tokenization
input_tokenizer = Tokenizer(filters='')
target_tokenizer = Tokenizer(filters='')
input_tokenizer.fit_on_texts(data['input_text'])
target_tokenizer.fit_on_texts(data['target_text'])

input_sequences = input_tokenizer.texts_to_sequences(data['input_text'])
target_sequences = target_tokenizer.texts_to_sequences(data['target_text'])

max_encoder_seq_length = max(len(seq) for seq in input_sequences)
max_decoder_seq_length = max(len(seq) for seq in target_sequences)

encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences([seq[:-1] for seq in target_sequences], maxlen=max_decoder_seq_length - 1, padding='post')
decoder_target_data = pad_sequences([seq[1:] for seq in target_sequences], maxlen=max_decoder_seq_length - 1, padding='post')

# Vocabulary sizes
num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

# Model hyperparameters
embedding_dim = 256
latent_dim = 512

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_encoder_tokens, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(num_decoder_tokens, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model compilation
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

# Train the model
model.fit([encoder_input_data, decoder_input_data],
          np.expand_dims(decoder_target_data, -1),
          batch_size=64,
          epochs=30,
          validation_split=0.2)

# Save model
model.save("nmt_poetry_translation.h5")

# Inference model setup
# Encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb2 = Embedding(num_decoder_tokens, embedding_dim)(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_states2 = [state_h2, state_c2]
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

# Reverse lookup for words
reverse_target_word_index = dict((i, word) for word, i in target_tokenizer.word_index.items())
reverse_target_word_index[0] = ''

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['<start>']
    decoded_sentence = ''
    stop_condition = False

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

# Try it out
def translate_line(input_line):
    seq = input_tokenizer.texts_to_sequences([input_line])
    padded_seq = pad_sequences(seq, maxlen=max_encoder_seq_length, padding='post')
    return decode_sequence(padded_seq)

if __name__ == "__main__":
    test_line = "चाँदनी रात में खामोशी बसी होती है"
    print("Hindi:", test_line)
    print("English Translation:", translate_line(test_line))
