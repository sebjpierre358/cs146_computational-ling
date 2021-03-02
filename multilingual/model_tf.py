import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size, bpe):
		super(RNN_Seq2Seq, self).__init__()
		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size

		#Define batch size and optimizer/learning rate
		self.batch_size = 100
		self.embedding_size = 100
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=.02)
		self.bpe = bpe

		self.bpe_embed = tf.keras.layers.Embedding(self.french_vocab_size, self.embedding_size)

		self.frn_embed = tf.keras.layers.Embedding(self.french_vocab_size, self.embedding_size)

		self.encode_W = tf.keras.layers.GRU(self.french_window_size, recurrent_dropout=0,
                   bias_initializer='truncated_normal', return_sequences=True, return_state=True)

		self.eng_embed = tf.keras.layers.Embedding(self.english_vocab_size, self.embedding_size)

		self.decode_W = tf.keras.layers.GRU(self.english_window_size, recurrent_dropout=0,
                   bias_initializer='truncated_normal', return_sequences=True)

		self.dense_1 = tf.keras.layers.Dense(256, activation='relu', bias_initializer='truncated_normal')
		self.dense_2 = tf.keras.layers.Dense(64, activation='relu', bias_initializer='truncated_normal')
		self.dense_3 = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax', bias_initializer='truncated_normal')

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
		frn_embed = self.bpe_embed(encoder_input) if self.bpe else self.eng_embed(frn_embed)
		encode_out, encode_final_state = self.encode_W(frn_embed)
		eng_embed = self.bpe_embed(decoder_input) if self.bpe else self.eng_embed(eng_embed)
		#input to the first cell here should be "STOP"
		context_decode = tf.concat([encode_out, eng_embed], 2)
		#decode_seq_out = self.decode_W(eng_embed, initial_state=encode_final_state)
		decode_seq_out = self.decode_W(context_decode, initial_state=encode_final_state)

		d1_out = self.dense_1(decode_seq_out)
		d2_out = self.dense_2(d1_out)
		prbs = self.dense_3(d2_out)

		#check prbs is shape [batch_size x window_size x english_vocab_size]
		return prbs

	def accuracy_function(self, prbs, labels, mask):
		"""
		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the total model cross-entropy loss after one forward pass.
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		masked_labels = tf.math.multiply(labels, mask)
		#not always guaranteed a full batch
		num_examps = labels.shape[0]
		loss = tf.keras.losses.sparse_categorical_crossentropy(masked_labels, prbs, from_logits=False)
		sentence_loss_sum = tf.math.reduce_sum(loss, axis=1)
		batch_loss_sum = tf.math.reduce_sum(sentence_loss_sum)
		return batch_loss_sum
