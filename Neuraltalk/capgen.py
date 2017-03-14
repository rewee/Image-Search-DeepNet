# download caffe model from http://www.robots.ox.ac.uk/~vgg/research/very_deep/
# download checkpoint model from http://cs.stanford.edu/people/karpathy/neuraltalk/

import caffe
import os.path
import numpy as np
from scipy.misc import imread, imresize
from neuraltalk.imagernn.imagernn_utils import decodeGenerator
import cPickle as pickle
### Global variable declarations 
caffe.set_mode_gpu()
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CAFFE_MODEL_DEF_PATH = os.path.join(FILE_DIR, 'python_features/deploy_features.prototxt')
CAFFE_MODEL_PATH = os.path.join(FILE_DIR, 'python_features/VGG_ILSVRC_16_layers.caffemodel')
CHECKPOINT_PATH = os.path.join(FILE_DIR, 'model/model_checkpoint_coco_visionlab43.stanford.edu_lstm_11.14.p')
BEAM_SIZE = 2
### CaptionGenerator Class
class CaptionGenerator:
    def __init__(self):
        '''
        Instantiates the caption generator class
        '''
        self.net = caffe.Net(CAFFE_MODEL_DEF_PATH, CAFFE_MODEL_PATH, caffe.TEST)
        self.checkpoint = pickle.load(open(CHECKPOINT_PATH, 'rb'))
        self.checkpoint_params = self.checkpoint['params']
        self.model = self.checkpoint['model']
        self.ixtoword = self.checkpoint['ixtoword']

    def extract_features_vgg16(iname):
        '''
        Given image, extracts the top layer from VGG net 
        '''
        img = caffe.io.load_image(iname)
        img = img[:,:,::-1]*255.0
        avg = np.array([103.939,116.779,123.68])
        img = img - avg 
        im = imresize(img,(224,244),'bicubic')
        im = np.transpose(im,(2,0,1))
        im=im[None,:]
        out = self.net.foward_all(data=im)
        features = out[net.outputs[0]]
        return features

    def predict(self, features):
        ''' Given the features, feeds them to the lstm to get descriptions
        '''
        BatchGenerator = decodeGenerator(CHECKPOINT_PATH)
        img = {}
        img['feat'] = features[:, 0]
        kwparams = {'beam_size': BEAM_SIZE}
        Ys = BatchGenerator.predict([{'image': img}], self.model, self.checkpoint_params, **kwparams)
        top_predictions = Ys[0]  # take predictions for the first (and only) image we passed in
        top_prediction = top_predictions[0]  # these are sorted with highest on top      
        candidate = ' '.join([self.ixtoword[ix] for ix in top_prediction[1] if ix > 0])
        return candidate

    # absolute file path
    def get_caption(self, file):
        allftrs = self.extract_features_vgg16(file)
        features = np.transpose(allftrs)
        caption = self.predict(features)
        return caption
