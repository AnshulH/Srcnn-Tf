import tensorflow as tf
from  model import SRCNN

flags = tf.app.flags

flags.DEFINE_integer('epoch', 1000, 'Number of epoch')
flags.DEFINE_integer('image_dim', 33, 'The size of image input')
flags.DEFINE_integer('label_dim', 21, 'The size of image output')
flags.DEFINE_integer('channel', 3, 'The size of channel')
flags.DEFINE_boolean('train', True, 'if the train')
flags.DEFINE_integer('scale', 3, 'the size of scale factor')
flags.DEFINE_integer('stride', 21, 'the size of stride')
flags.DEFINE_string('chkpt', 'Checkpoint', 'Name of checkpoint directory')
flags.DEFINE_float('learning_rate', 1e-4 , 'The learning rate')
flags.DEFINE_integer('batch_size', 64, 'the size of batch')
flags.DEFINE_string('result_dir', 'Result', 'Name of result directory')

FLAGS = flags.FLAGS


def main(_): 
    with tf.Session() as sess:
        srcnn = SRCNN(sess,
                      image_dim = FLAGS.image_dim,
                      label_dim = FLAGS.label_dim,
                      channel = FLAGS.channel)

        srcnn.train(FLAGS)

if __name__=='__main__':
    tf.app.run() 