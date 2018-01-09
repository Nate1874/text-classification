from __future__ import absolute_import, division, print_function
import argparse
import tensorflow as tf
import time
import os
import sys
from model import Attention

def configure():
    flags = tf.app.flags
    flags.DEFINE_integer("batch_size", 64, "batch size")
    flags.DEFINE_integer("max_epoch", 1000, "max epoch for total training")
    flags.DEFINE_integer("sequence_length", 21273, "sequence_length")
    flags.DEFINE_integer("num_classes", 20, "num_classes")
    flags.DEFINE_integer("embed_size", 300, "embed_size")    
    flags.DEFINE_integer("vocab_size", 131739, "vocab size")
    flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many iterations")
    flags.DEFINE_integer("summary_step", 100, "save summary per #summary_step iters")
    flags.DEFINE_integer("save_step", 100, "save model per #save_step iters")
    flags.DEFINE_integer("eva_step", 2, "save model per #save_step iters")
    flags.DEFINE_integer("n_class", 10, "number of classes")
    flags.DEFINE_float("learning_rate", 2e-4, "learning rate")
    flags.DEFINE_boolean("enable_word_embeddings", True, "if use pre-trained embedding")
    flags.DEFINE_float("gan_noise", 0.01, "injection noise for the GAN")
    flags.DEFINE_bool("noise_bool", False, "add noise on all GAN layers")
    flags.DEFINE_string("working_directory", "/tempspace/hyuan/cell_aae",
                        "the file directory")
    flags.DEFINE_integer("hidden_size", 16, "size of the hidden VAE unit")
    flags.DEFINE_integer("checkpoint", 0, "number of epochs to be reloaded")
    flags.DEFINE_integer("d_model", 512, "dimension in the model")
    flags.DEFINE_integer("d_k", 64, "dimension of key")
    flags.DEFINE_integer("d_v", 64, "dimension of value")
    flags.DEFINE_integer("h", 8, "number of heads")
    flags.DEFINE_string("modeldir", './modeldir', "the model directory")
    flags.DEFINE_string("logdir", './logdir', "the log directory")
    flags.DEFINE_string("sampledir", './sampledir', "the sample directory")
    flags.DEFINE_string("path_embedding", '/tempspace/hyuan/data_text/GoogleNews-vectors-negative300.bin', "the path for pre-trained embedding")
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS
 
def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--action',
        dest='action',
        type=str,
        default='train',
        help='actions: train, or test')
    args = parser.parse_args()
    if args.action not in ['train', 'test']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test")
    else:
        model= Attention(tf.Session(),configure())
        getattr(model,args.action)()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    tf.app.run()





