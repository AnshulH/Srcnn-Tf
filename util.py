import os
import cv2
from scipy import misc
import numpy as np
import h5py
import tensorflow as tf
import glob

FLAGS = tf.app.flags.FLAGS
'''
scale = 3.0
inpSize = 33
labelSize = 21
stride = 14 
path = "Images/"
paths = os.listdir(path)
paths_length = len(paths)  
'''

def read_image(path):

  image = cv2.imread(path,cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb).astype(np.float)
  return image
  
def prep(sess,dataset):
  if FLAGS.train:
    files = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(),dataset)
    data = glob.glob(os.path.join(data_dir,'*.bmp'))
  else:
    data_dir = os.path.join(os.sep,(os.path.join(os.getcwd(),dataset)))
    print(data_dir)
    data = glob.glob(os.path.join(data_dir,'*.bmp'))
  return data  

def read_data(path):
  with h5py.File(path,'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
  return data,label

def create_data(sess,data,label):

  if not os.path.exists(os.path.join(os.getcwd(),'checkpoint')):
    os.makedirs(os.path.join(os.getcwd(),'checkpoint'))
  if FLAGS.train:
    save = os.path.join(os.getcwd(),'checkpoint\\train.h5')
  else:
    save = os.path.join(os.getcwd(),'checkpoint\\test.h5')  

  with h5py.File(save,'w') as hf:
    hf.create_dataset('data',data=data)
    hf.create_dataset('label',data=label)

def preprocess(path,scale=3):

  image = read_image(path)
  #print(image.shape)
  
  h,w,c = image.shape
  h = h - h%scale
  w = w - w%scale
  label_prep = image[0:h,0:w,:]


  image = image / 255.0
  label_prep = label_prep / 255.0

  bicubic_img = cv2.resize(label_prep,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC)
  scaled_image = cv2.resize(bicubic_img,None,fx = scale ,fy=scale, interpolation = cv2.INTER_CUBIC)
  scaled_label = label_prep

  return scaled_image, scaled_label

def input_setup(sess,config):
  
  data_inp_list = []
  data_lab_list = []   


  if config.train:
    directory = 'Data' + os.sep + 'Train' 
    data = prep(sess,dataset=str(directory))
    print(data[0])
  else:
    directory = 'Data' + os.sep + 'Test'
    data = prep(sess,dataset=str(directory))


  pad = abs(config.image_dim - config.label_dim) / 2

  if config.train:

    for i in range(0,len(data)):
      inp, lab = preprocess(data[i],config.scale)      

    height, width, channel = inp.shape

    for x in range(0,height-config.image_dim+1, config.stride):
      for y in range(0,width-config.image_dim+1, config.stride):
        sub_input = inp[x:x+config.image_dim, y:y+config.image_dim]
        sub_label = lab[x+int(pad):x+int(pad)+config.label_dim, y+int(pad):y+int(pad)+config.label_dim]
        # print(sub_label.shape)

        sub_input = sub_input.reshape([config.image_dim, config.image_dim, config.channel])
        sub_label = sub_label.reshape([config.label_dim, config.label_dim, config.channel])

        data_inp_list.append(sub_input)
        data_lab_list.append(sub_label)

  else:
    inp, lab = preprocess(data[2], config.scale)

    height, width, channel = inp.shape

    sub_img_countx = 0
    sub_img_county = 0

    for x in range(0,height-config.image_dim+1,config.stride):
      sub_img_countx += 1 
      sub_img_county = 0  
      for y in range(0,width-config.image_dim+1,config.stride):
        sub_img_county += 1
        sub_input = inp[x:x+config.image_dim,y:y+config.image_dim]
        sub_label = lab[x+int(pad):x+int(pad)+config.label_dim,y+int(pad):y+int(pad)+config.label_dim]

        sub_input = sub_input.reshape([config.image_dim,config.image_dim,config.channel])
        sub_label = sub_label.reshape([config.label_dim,config.label_dim,config.channel])

        data_inp_list.append(sub_input)
        data_lab_list.append(sub_label)


  arr_inp = np.asarray(data_inp_list)
  arr_lab = np.asarray(data_lab_list)

  create_data(sess,arr_inp,arr_lab)

  if not config.train:
    return sub_img_countx,sub_img_county

def save_img(image,path,config):
  if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
    os.makedirs(os.path.join(os.getcwd(),config.result_dir))
  
    cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)

def merge(images, size, channel):
  h, w = images.shape[1], images.shape[2]
  image = np.zeros((h*size[0], w*size[1], channel))
  for idx, img in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    image[j*h:j*h+h,i*w:i*w+w,:] = img
    return image
'''
print(paths)
for x in range(paths_length):
  image_path = path + paths[x]
  print(image_path)
  image = cv2.imread(image_path ,cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image ,cv2.COLOR_BGR2YCrCb).astype(np.float)
    
  w, h, c = image.shape
  w = w - w%3
  h = h - h%3
  image = image[0:w,0:h,:]

  scaled = misc.imresize(image , 1.0/scale, 'bicubic')
  scaled = misc.imresize(scaled , scale/1.0 , 'bicubic')
  no = 1
  for i in range(0 , h-inpSize+1,stride):
    for j in range(0 , w-inpSize+1,stride):
      dataInp = scaled[j:j+inpSize,i:i+inpSize]
      dataLabel = image[j + 6 : j + 6 + labelSize , i + 6 : i + 6 + labelSize]
      dataInp = dataInp.reshape([inpSize,inpSize,1])
      dataLabel = dataLabel.reshape([labelSize,labelSize,1])
      dataInp_list.append(dataInp)
      dataLabel_list.append(dataLabel)
      cv2.imwrite(os.path.join('Input/Inp'+ str(no) + '.bmp'), dataInp)
      cv2.imwrite(os.path.join('Label/Label' + str(no) + '.bmp'), dataLabel)
      no += 1

save = os.path.join(os.getcwd(), 'checkpoint/train.h5')

data = np.asarray(dataInp_list)
label = np.asarray(dataLabel_list)

with h5py.File(save,'w') as hf:
  hf.create_dataset('data',data=data)
  hf.create_dataset('label',data=label)
 ''' 
