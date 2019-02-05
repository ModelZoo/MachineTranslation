from model import Encoder, DecoderWithAttention
import tensorflow as tf

if __name__ == '__main__':
    config = {
        'batch_size': 64,
        'embedding_size': 300,
        'vocab_size': 10000,
        'hidden_units': 100,
        'max_length': 25,
        'attention_units': 200,
    }
    encoder = Encoder(config)
    decoder = DecoderWithAttention(config)
    print(encoder)
    encoder_inputs = tf.random_normal(shape=[config['batch_size'], config['max_length']])
    print('encoder_inputs', encoder_inputs)
    encoder_outputs, state = encoder(encoder_inputs)
    print(encoder_outputs, state)
    decoder_inputs = tf.random_normal(shape=[config['batch_size'], 1])
    outputs, state = decoder(decoder_inputs, state, encoder_outputs)
    print(outputs, state)
