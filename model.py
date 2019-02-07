from model_zoo.model import BaseModel
import tensorflow as tf
import numpy as np


def gru(units):
    """
    use CuDNNGRU for GPU, otherwise normal GRU
    :param units:
    :return:
    """
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
    return tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
    """
    Simple encoder based on GRU
    """
    
    def __init__(self, config):
        """
        Initialize all variables
        :param config: args
        """
        super(Encoder, self).__init__()
        self.batch_size = config['batch_size']
        self.embedding_size = config['embedding_size']
        self.vocab_size = config['vocab_size']
        self.hidden_units = config['hidden_units']
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.gru = gru(self.hidden_units)
    
    def call(self, inputs, state=None):
        """
        Encode all texts to outputs and final state
        :param inputs: shape: [batch_size, max_length, hidden_units]
        :param state: shape: [batch_size, hidden_units]
        :return:
        """
        inputs = self.embedding(inputs)
        state = state if state else self.zero_state
        return self.gru(inputs, initial_state=state)
    
    @property
    def zero_state(self):
        """
        Get zero state
        :return:
        """
        return tf.zeros((self.batch_size, self.hidden_units))


class Decoder(tf.keras.Model):
    
    def __init__(self, config):
        """
        Initialize all variables
        :param config: args
        """
        super(Decoder, self).__init__()
        self.batch_size = config['batch_size']
        self.embedding_size = config['embedding_size']
        self.vocab_size = config['vocab_size']
        self.hidden_units = config['hidden_units']
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        # gru
        self.gru = gru(self.hidden_units)
        
        # dense for vocab transform
        self.dense = tf.keras.layers.Dense(self.vocab_size)
    
    def call(self, inputs, state):
        """
        Define base decoder for seq2seq model.
        :param inputs:
        :param state:
        :return:
        """
        # inputs: [batch_size, 1, embedding_size]
        inputs = self.embedding(inputs)
        # state: [batch_size, hidden_units]
        outputs, state = self.gru(inputs, initial_state=state)
        # outputs: [batch_size, hidden_units]
        outputs = self.dense(tf.reshape(outputs, [-1, outputs.shape[-1]]))
        return outputs, state
    
    @property
    def zero_state(self):
        """
        Get zero state
        :return:
        """
        return tf.zeros((self.batch_size, self.hidden_units))


class DecoderWithAttention(tf.keras.Model):
    """
    Decoder with Attention
    """
    
    def __init__(self, config):
        """
        Initialize all variables.
        :param config: args
        """
        super(DecoderWithAttention, self).__init__()
        self.batch_size = config['batch_size']
        self.embedding_size = config['embedding_size']
        self.vocab_size = config['vocab_size']
        self.hidden_units = config['hidden_units']
        self.attention_units = config['attention_units']
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.gru = gru(self.hidden_units)
        # dense for attention
        self.dense_w = tf.keras.layers.Dense(self.attention_units)
        self.dense_u = tf.keras.layers.Dense(self.attention_units)
        self.dense_v = tf.keras.layers.Dense(1)
        # dense for vocab transform
        self.dense = tf.keras.layers.Dense(self.vocab_size)
    
    def call(self, inputs, state, encoder_outputs):
        """
        Process decoder step with attention mechanism.
        :param inputs:
        :param state:
        :param encoder_outputs:
        :return:
        """
        # e_i: [batch_size, max_length]
        e_i = self.dense_v(tf.nn.tanh(self.dense_w(tf.expand_dims(state, 1)) + self.dense_u(encoder_outputs)))
        # alpha_i: [batch_size, max_length, 1]
        alpha_i = tf.nn.softmax(e_i, axis=1)
        # c_i: [batch_size, hidden_units, 1]
        c_i = tf.reduce_sum(alpha_i * encoder_outputs, axis=1)
        # inputs: [batch_size, 1, embedding_size]
        inputs = self.embedding(inputs)
        # inputs: [batch_size, 1, embedding_size + hidden_units]
        inputs = tf.concat([tf.expand_dims(c_i, 1), inputs], axis=-1)
        # outputs: [batch_size, 1, hidden_units]
        # state: [batch_size, hidden_units]
        outputs, state = self.gru(inputs, initial_state=state)
        # outputs: [batch_size, hidden_units]
        outputs = self.dense(tf.reshape(outputs, [-1, outputs.shape[-1]]))
        return outputs, state
    
    @property
    def zero_state(self):
        """
        Get zero state
        :return:
        """
        return tf.zeros((self.batch_size, self.hidden_units))


class Seq2SeqModel(BaseModel):
    
    def __init__(self, config):
        """
        Init base encoder and decoder.
        :param config:
        """
        super(Seq2SeqModel, self).__init__(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def init(self):
        """
        Init model.
        :return:
        """
        self.compile(optimizer=self.optimizer(),
                     loss=self.loss,
                     metrics=[self.metric])
    
    def loss(self, y_true, y_pred):
        """
        Define loss function.
        :param y_true: label
        :param y_pred: logits
        :return:
        """
        y_true = y_true[:, 1:]
        mask = 1 - np.equal(y_true, 0)
        loss_matrix = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) * mask
        loss_batch = tf.reduce_sum(loss_matrix, axis=-1)
        loss = tf.reduce_mean(loss_batch)
        return loss
    
    def metric(self, y_true, y_pred):
        """
        Define metric for loss.
        :param y_true:
        :param y_pred:
        :return:
        """
        y_true = y_true[:, 1:]
        mask = 1 - np.equal(y_true, 0)
        loss_matrix = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) * mask
        loss_batch = tf.reduce_sum(loss_matrix, axis=-1)
        loss = tf.reduce_mean(loss_batch)
        return loss
    
    def call(self, inputs, training=None, mask=None):
        """
        Run seq2seq model
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        sources, targets = inputs
        encoder_outputs, state = self.encoder(sources)
        
        if training:
            decoder_outputs, decoder_states = [], []
            for i in range(tf.shape(sources)[-1] - 1):
                source = tf.expand_dims(sources[:, i], 1)
                outputs, state = self.decoder(source, state)
                decoder_outputs.append(outputs)
                decoder_states.append(state)
            decoder_outputs = tf.stack(decoder_outputs, axis=1)
            return decoder_outputs


class Seq2SeqAttentionModel(Seq2SeqModel):
    
    def __init__(self, config):
        """
        Init encoder and attention-decoder, define input and output shape.
        :param config:
        """
        super(Seq2SeqAttentionModel, self).__init__(config)
        self.encoder = Encoder(config)
        self.decoder = DecoderWithAttention(config)
        self.shape(inputs_shape=[config['max_length']], output_shape=[config['vocab_size']])
    
    def call(self, inputs, training=None, mask=None):
        """
        Run seq2seq attention model.
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        sources, targets = inputs
        encoder_outputs, state = self.encoder(sources)
        
        if training:
            # define decode process
            decoder_outputs, decoder_states = [], []
            for i in range(tf.shape(sources)[-1] - 1):
                source = tf.expand_dims(sources[:, i], 1)
                outputs, state = self.decoder(source, state, encoder_outputs)
                decoder_outputs.append(outputs)
                decoder_states.append(state)
            decoder_outputs = tf.stack(decoder_outputs, axis=1)
            return decoder_outputs
