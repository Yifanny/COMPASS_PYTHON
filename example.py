import math
import numpy as np
import os
import struct
import sys

from sklearn.linear_model import SGDClassifier

def load_mnist(path, kind='train'):
	"""Load MNIST data from `path`"""
	labels_path = os.path.join(path,'%s-labels.idx1-ubyte'%kind)
	images_path = os.path.join(path,'%s-images.idx3-ubyte'%kind)
	
	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II', lbpath.read(8))
		labels = np.fromfile(lbpath, dtype=np.uint8)
		
	with open(images_path, 'rb') as imgpath:
		magic, num, row, cols = struct.unpack('>IIII', imgpath.read(16))
		images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
	
	return images, labels
	
def show_image(image, label):
	plt.figure(figsize=(20,4))
	for index, (img, lab) in enumerate(zip(image[0:5], label[0:5])):
		plt.subplot(1, 5, index+1)
		plt.imshow(np.reshape(img, (28, 28)), cmap=plt.cm.gray)
		plt.title(lab, fontsize = 20)
	plt.show()
	return 0

def mnist_learn(train_img, train_label, l_rate, rglar, batch_s):
    batch_s = int(batch_s)
    #epoch = int(epoch)
    #select_id = random.sample(range(60000), batch_s)
    select_id = range(batch_s)
    select_img = train_img[select_id[0]]
    select_label = train_label[select_id[0]]
    for i in range(batch_s-1):
		#print(i)
		#print(select_img.shape)
        select_img = np.vstack((select_img, train_img[select_id[i+1]]))
        select_label = np.vstack((select_label, train_label[select_id[i+1]]))
	
    #clf = SGDClassifier(alpha = rglar, learning_rate = 'constant', eta0 = l_rate, max_iter = epoch, tol = None)
    clf = SGDClassifier(alpha = rglar, learning_rate = 'constant', eta0 = l_rate)
	#show_image(select_img, select_label)
	#clf = SGDClassifier()
	#print(select_img.shape)
    clf.fit(select_img, np.ravel(select_label))
    
    return clf
	
def mnist_classifier(clf, test_img, test_label):
	label_pred = clf.predict(test_img)
	acc = clf.score(test_img, test_label)
	
	return label_pred, acc 

def mnist(params):
	learning_rate = params[0]
	regularization = params[1]
	
	batch_size = 2000
	images, labels = load_mnist('/Users/zhangyifan/Documents/RA_pre/bayesian_opt/Spearmint-master_new/examples/mnist/data')
	#show_image(images, labels)
	clf = mnist_learn(images, labels, learning_rate, regularization, batch_size)
	test_images, test_labels = load_mnist('/Users/zhangyifan/Documents/RA_pre/bayesian_opt/Spearmint-master_new/examples/mnist/data', 't10k')
	label_pred, acc = mnist_classifier(clf, test_images, test_labels)
	err = 1-acc
    #print('Result = %f' %err)
	return err