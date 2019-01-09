import tensorflow as tf
import cv2 as cv2
import numpy as np
import os.path
import os
from PIL import Image

from util import(input_setup,
				  merge,
				  save_img,
				  create_data,
				  read_data)
	

class SRCNN(object):

	def __init__(self,
		sess,
		image_dim = 33,
		label_dim = 21,
		channel = 3,
		chkpt = 'Checkpoint'):

		self.sess = sess
		self.image_dim = image_dim
		self.label_dim = label_dim
		self.channel = 3
		self.chkpt = chkpt

		self.weights = {}
		self.biases = {}

		self.images = tf.placeholder(tf.float32, [None,self.image_dim,self.image_dim,self.channel],name = 'images')
		self.labels = tf.placeholder(tf.float32, [None,self.label_dim,self.label_dim,self.channel],name = 'labels')

		self.model_init()
			
	def conv2d_op(self,inputs,filter_size,strides=[1,1,1,1],name=None,trainable=True):
		filters = self.get_filters(filter_size,name,trainable)
		strides = [1,1,1,1]

		conv_layer = tf.nn.conv2d(inputs,filters,strides = strides,padding='VALID',name=name + '_layer')
		conv_layer = tf.add(conv_layer,self.bias_add(filter_size[-1],name))
		if name != 'l3_layer':
			conv_layer = tf.nn.relu(conv_layer)

		return conv_layer

	def bias_add(self,bias_size,name,trainable=True):
		name = name + '_bias'
		bias = tf.Variable(tf.zeros([bias_size]),name=name)
		self.biases[name] = bias

		return bias	

	def get_filters(self,filter_size,name,trainable=True):
		name = name + '_weights'
		conv_weights = tf.Variable(tf.random_normal(filter_size,stddev=0.001),name=name,trainable=trainable)
		#conv_weights = tf.add(conv_weights,self.bias_add(filter_size[-1],name))
		self.weights[name] = conv_weights

		return conv_weights		

	def model(self):
		
		conv1 = self.conv2d_op(self.images, filter_size=[9,9,self.channel,64], name='l1')

		conv2 = self.conv2d_op(conv1, filter_size=[1,1,64,32], name='l2')

		conv3 = self.conv2d_op(conv2, filter_size=[5,5,32,1], name='l3')

		return conv3

	def model_init(self):
		
		self.pred = self.model()

		self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))	

		self.saver = tf.train.Saver()

	def save(self,chkpt,step):
		model_name = 'model'
		model_dir = os.path.join(chkpt+os.sep+'Model_save')
		print(model_dir)
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

		self.saver.save(self.sess,os.path.join(model_dir,model_name),global_step=step)		

	def load(self,chkpt):
		model_name = 'model'
		model_dir = os.path.join(chkpt+os.sep+'model_save')
		#print(model_dir)
		ckpt = tf.train.get_checkpoint_state(chkpt)

		if ckpt and ckpt.model_checkpoint_path:
			ckpt_path = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess,os.path.join(chkpt,ckpt_path))
			print('Success')
			return True
		else:
			print('Failure')
			return False	

	def train(self,config):

		if config.train:
			input_setup(self.sess,config)
			data = os.path.join(os.getcwd(),'checkpoint\\train.h5')
			print(data)
			 
		else:
			x,y = input_setup(self.sess,config)
			data = os.path.join(os.getcwd(),'checkpoint\\test.h5')
				
		train_data,train_label = read_data(data)	
		#print(train_label.shape)

		self.optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)

		self.sess.run(tf.global_variables_initializer())

		counter = 0

		if self.load(self.chkpt):
			print('Load Success')
		else:
			print('Load fail')

		if config.train:		
			for i in range(config.epoch):
				batch_idx = len(train_data) // config.batch_size
				for idx in range(0,batch_idx):
					batch_data = train_data[idx * config.batch_size:(idx+1)* config.batch_size]
					batch_label = train_label[idx * config.batch_size:(idx+1)* config.batch_size]
					#print(train_data.shape)

					counter +=1

					_,out = self.sess.run([self.optimizer,self.loss],feed_dict={self.images:batch_data , self.labels:batch_label})

					if counter%10 == 0:
						print("epoch:%d, step:%d, loss:%.8f" % (i,counter,out))

					if counter%100 == 0:
						print(os.path.join(self.chkpt,"model_save"))
						self.saver.save(self.sess,os.path.join(self.chkpt,"model_save"),global_step=counter)

		else:
			out = self.pred.eval({self.images:train_data,self.labels:train_label })

			image = merge(out, [x,y], config.channel)
			
			image_path = os.path.join(os.getcwd(), config.result_dir)
			image_path = os.path.join(image_path, "test_image.png")
			#Image.open(image).show()
			save_img(image,image_path,config)

'''
	def model_init(self):
		
		self.weights = {
			'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
			'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
      		'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    	}#for
		#self.weights = tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2')
		self.biases = {
     		'b1': tf.Variable(tf.zeros([64]), name='b1'),
      		'b2': tf.Variable(tf.zeros([32]), name='b2'),
      		'b3': tf.Variable(tf.zeros([1]), name='b3')
    	}

	def model() 
		self.pred = self.model()

		self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
'''