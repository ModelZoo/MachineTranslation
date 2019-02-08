import tensorflow as tf
from sklearn.model_selection import train_test_split
from model_zoo.trainer import BaseTrainer
from os.path import join
import pickle

tf.flags.DEFINE_integer('epochs', 20, 'Max epochs')
tf.flags.DEFINE_integer('embedding_size', 300, 'Embedding size')
tf.flags.DEFINE_integer('batch_size', 16, 'Batch size')
tf.flags.DEFINE_integer('vocab_size', 10000, 'Vocabulary size')
tf.flags.DEFINE_integer('attention_units', 250, 'Attention units')
tf.flags.DEFINE_integer('hidden_units', 200, 'Hidden units')
tf.flags.DEFINE_integer('max_length', 20, 'Max length')
tf.flags.DEFINE_string('model_class', 'Seq2SeqAttentionModel', 'Model class name')
tf.flags.DEFINE_string('datasets_dir', './datasets', help='Data dir')
tf.flags.DEFINE_string('dataset', 'es2en', help='Data dir')
tf.flags.DEFINE_bool('multiple_inputs', True, 'Multiple inputs')
tf.flags.DEFINE_string('checkpoint_dir', 'checkpoints/es2en', help='Data source dir')


class Trainer(BaseTrainer):
    
    def prepare_data(self):
        sources = pickle.load(open(join(self.flags.datasets_dir, self.flags.dataset, 'sources.pkl'), 'rb'))
        targets = pickle.load(open(join(self.flags.datasets_dir, self.flags.dataset, 'targets.pkl'), 'rb'))
        sources_train, sources_eval, targets_train, targets_eval = train_test_split(sources, targets, test_size=0.1)
        return ([sources_train, targets_train], targets_train), ((sources_eval, targets_eval), targets_eval)


if __name__ == '__main__':
    Trainer().run()
