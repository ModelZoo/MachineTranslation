from os.path import join
from sklearn.model_selection import train_test_split
import json, pickle
from model_zoo.inferer import BaseInferer
import tensorflow as tf
import numpy as np

tf.flags.DEFINE_string('checkpoint_name', 'model.ckpt', help='Model name')
tf.flags.DEFINE_integer('vocab_size', 10000, help='Vocab size')
tf.flags.DEFINE_string('checkpoint_dir', 'checkpoints/es2en', help='Data source dir')
tf.flags.DEFINE_string('datasets_dir', './datasets', help='Data dir')
tf.flags.DEFINE_string('dataset', 'es2en', help='Data dir')


class Inferer(BaseInferer):
    
    def seq2str(self, seq, type='target'):
        """
        Transfer seq to string
        :param seq:
        :return:
        """
        seq = seq.tolist()
        vocab = self.targets_vocab if type == 'target' else self.sources_vocab
        result = ''
        for s in seq:
            if s == 4: break
            if s == 5: continue
            result += vocab[str(s)] + ' '
        return result.strip()
    
    def prepare_data(self):
        """
        Main prepare data
        :return:
        """
        self.sources_vocab = json.load(
            open(join(self.flags.datasets_dir, self.flags.dataset, 'sources_vocab.json'), 'r'))
        self.targets_vocab = json.load(
            open(join(self.flags.datasets_dir, self.flags.dataset, 'targets_vocab.json'), 'r'))
        sources = pickle.load(open(join(self.flags.datasets_dir, self.flags.dataset, 'sources.pkl'), 'rb'))
        targets = pickle.load(open(join(self.flags.datasets_dir, self.flags.dataset, 'targets.pkl'), 'rb'))
        _, sources_test, _, targets_test = train_test_split(sources, targets, test_size=0.05, random_state=10)
        self.sources_test = sources_test
        print('sources_test', sources_test.shape, 'targets_test', targets_test.shape)
        return sources_test, targets_test


if __name__ == '__main__':
    inf = Inferer()
    logits = inf.run()
    for source, logit in zip(inf.sources_test, logits):
        predict = np.argmax(logit, axis=1)
        result_source = inf.seq2str(source, 'source')
        result_target = inf.seq2str(predict, 'target')
        print('=' * 20)
        print(result_source)
        print(result_target)
