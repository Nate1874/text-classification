import numpy as np
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files

class data_reader:
    def __init__(self):
        print("Now loading data ==================")
        self.path_pos= '/tempspace/hyuan/data_text/rt-polarity.pos'
        self.path_neg= '/tempspace/hyuan/data_text/rt-polarity.neg'
   #    datasets = self.get_datasets_polarity(self.path_pos, self.path_neg)
        self.datasets = self.get_datasets_20newsgroup(shuffle=True, random_state=1)
        self.x_text, y = self.load_data_labels_news(self.datasets)
        self.max_document_length = max([len(x.split(" ")) for x in self.x_text])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_document_length)
        x = np.array(list(self.vocab_processor.fit_transform(self.x_text)))    
        ## Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        dev_sample_index = -1 * int(0.1 * float(len(y)))  # use 10% data for testing
        self.x_train, self.x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        self.y_train, self.y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("Vocabulary Size: {:d}".format(len(self.vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(self.y_train), len(self.y_dev)))    
        print("Finished loading data ==================")
        self.gen_index()
        self.train_idx = 0 
        self.test_idx = 0

    def get_datasets_20newsgroup(self, shuffle=True, random_state=42):

        datasets = fetch_20newsgroups(shuffle=shuffle, random_state=random_state)
        return datasets

    def gen_index(self):
        self.indexes = np.random.permutation(range(len(self.y_train)))
        self.train_idx = 0

    def next_batch(self, batch_size):
        next_index = self.train_idx + batch_size
        cur_indexes = list(self.indexes[self.train_idx:next_index])
        self.train_idx = next_index
        if len(cur_indexes) < batch_size:
            self.gen_index()
            return self.next_batch(batch_size)
        cur_indexes.sort()
        return self.x_train[cur_indexes], self.y_train[cur_indexes]   
    
    def next_test_batch(self,batch_size):
        prev_idx = self.test_idx
        self.test_idx += batch_size
        return self.x_dev[prev_idx:self.test_idx], self.y_dev[prev_idx:self.test_idx]
    
    def reset():
        self.test_idx = 0

    def clean_str(self,string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def get_datasets_polarity(self, pos_file, neg_file):
        '''
        load the polarity dataset from two separate files.
        '''
        positive_examples = list(open(pos_file, "r", encoding='latin-1').readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(neg_file, "r", encoding='latin-1').readlines())
        negative_examples = [s.strip() for s in negative_examples]

        datasets = dict()
        datasets['data'] = positive_examples + negative_examples
        target = [0 for x in positive_examples] + [1 for x in negative_examples]
        datasets['target'] = target
        datasets['target_names'] = ['positive_examples', 'negative_examples']
        return datasets

    def load_data_labels_news(self, datasets):
        x_text = datasets.data
        x_text = [self.clean_str(sent) for sent in x_text]
        # labels = []
        # for i in range(len(x_text)):
        #     label = [0 for j in datasets['target_names']]
        #     label[datasets['target'][i]] = 1
        #     labels.append(label)
        # y = np.array(labels)
        return [x_text, datasets.target]   

    def load_data_labels(self, datasets):
        x_text = datasets['data']
        x_text = [self.clean_str(sent) for sent in x_text]
        labels = []
        for i in range(len(x_text)):
            label = [0 for j in datasets['target_names']]
            label[datasets['target'][i]] = 1
            labels.append(label)
        y = np.array(labels)
        return [x_text, y]    

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


    def load_word2vec_embedding(self, vocabulary, filename, binary):
        '''
        Load the word2vec embedding. 
        '''
        encoding = 'utf-8'
        with open(filename, "rb") as f:
            header = f.readline()
            vocab_size, vector_size = map(int, header.split())
            # initial matrix with random uniform
            embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
            if binary:
                binary_len = np.dtype('float32').itemsize * vector_size
                for line_no in range(vocab_size):
                    word = []
                    while True:
                        ch = f.read(1)
                        if ch == b' ':
                            break
                        if ch == b'':
                            raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                        if ch != b'\n':
                            word.append(ch)
                    word = str(b''.join(word), encoding=encoding, errors='strict')
                    idx = vocabulary.get(word)
                    if idx != 0:
                        embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                    else:
                        f.seek(binary_len, 1)
            else:
                for line_no in range(vocab_size):
                    line = f.readline()
                    if line == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                    word, vector = parts[0], list(map('float32', parts[1:]))
                    idx = vocabulary.get(word)
                    if idx != 0:
                        embedding_vectors[idx] = vector
            f.close()
            return embedding_vectors