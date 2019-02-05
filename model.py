from model_zoo.model import BaseModel
import tensorflow as tf


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
        x = self.embedding(inputs)
        state = state if state else self.zero_state
        output, state = self.gru(x, initial_state=state)
        return output, state
    
    @property
    def zero_state(self):
        """
        Get zero state
        :return:
        """
        return tf.zeros((self.batch_size, self.hidden_units))


if __name__ == '__main__':
    config = {
        'batch_size': 64,
        'embedding_size': 300,
        'vocab_size': 10000,
        'hidden_units': 100,
        'max_length': 25
    }
    encoder = Encoder(config)
    print(encoder)
    inputs = tf.random_normal(shape=[config['batch_size'], config['max_length']])
    print('inputs', inputs)
    output, state = encoder(inputs)
    
    print(output, state)


class Decoder():
    pass


class DecoderWithAttention(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(DecoderWithAttention, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, hidden_size)
        score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


class Seq2SeqModel(BaseModel):
    
    def __init__(self, config):
        super(Seq2SeqModel, self).__init__(config)
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, inputs, training=None, mask=None):
        o = self.dense(inputs)
        return o


class Seq2SeqAttentionModel(BaseModel):
    
    def __init__(self, config):
        super(Seq2SeqAttentionModel, self).__init__(config)
        self.encoder = Encoder()
        self.decoder = DecoderWithAttention()
    
    def call(self, inputs, training=None, mask=None):
        o = self.dense(inputs)
        return o
