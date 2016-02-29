import pickle
import sklearn
from sklearn import externals
from sklearn import datasets
from sklearn import linear_model
from sklearn import pipeline
from sklearn import neural_network
from sklearn import svm
from sklearn import metrics
from sklearn import cross_validation
from sklearn import grid_search
import scipy
from scipy import ndimage
import skimage
import numpy as np

def pickle_data(clf):
	externals.joblib.dump(clf, 'mnist_rbm_svc.pkl')

def load_data():
	clf = externals.joblib.load('mnist_rbm_svc.pkl')

def expand_data(X, Y):
	#we expand as the data with one raw pixel can give inaccurate results
	direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]
	shift = lambda x, w: ndimage.convolve(x.reshape((28, 28)), mode='constant', weights=w).ravel()
	
	X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors]) 
	Y = np.concatenate([Y for _ in range(5)], axis=0)
	return X, Y

def optimise(estimator):
	#find suitable hyperparameters for the given classifier in use
	#this method may take quite a time to run
	if estimator is 'svc' :
		#finding suitable hyperparameters
		param_grid_svc=[{'kernel':['rbf'],'gamma':[1e-3,1e-4], 'C':[1,10,100,1000]},
		{'kernel':['linear'],'C':[1,10,100,1000]}]
	
		#hyperparameter optimisation
		classifier_svc = grid_search.GridSearchCV(svm.SVC(C=1), param_grid_svc, cv=5, scoring='precision_weighted')
		classifier_svc.fit(i_train, l_train)

		#result is a dict - {'gamma': 0.001, 'kernel': 'rbf', 'C': 10} 
		print(classifier_svc.best_params_)
		return(classifier_svc.best_params_)

	else:
		#test for possible hyper parameters
		param_grid_rbm=[{'learning_rate':[0.1, 0.3, 0.45, 0.6, 0.75, 0.9], 'n_iter':[10,20,30,40], 
		'n_components':[100,200]}]

		#hyperparameter optimisation
		classifier_rbm = grid_search.GridSearchCV(neural_network.BernoulliRBM(), param_grid_rbm, cv=5, scoring='precision_weighted')
		classifier_rbm.fit(i_train, l_train)		

		#result (dict) - learning_rate = 0.01 n_iter = 40 n_components = 200
		print(classifier_svc.best_params_)
		return(classifier_svc.best_params_)		

def make_clf(i_train, l_train):
	#rbm is used for non linear feature extraction
	rbm_fe = neural_network.BernoulliRBM(random_state=0, verbose=True)

	#svc will be used for predicting
	svc_clf = svm.SVC()

	#create a pipeline of feature aggregator and the classifier
	classifier = pipeline.Pipeline(steps=[('rbm', rbm_fe), ('svc', svc_clf)])
	
	#setting the parameters
	rbm_fe.learning_rate = 0.06
	rbm_fe.n_iter = 30
	rbm_fe.n_components = 200
	svc_clf.C = 100
	svc_clf.gamma = 0.0325
	svc_clf.kernel = 'rbf'

	classifier.fit(i_train, l_train)

	return classifier

def predict_clf(classifier, i_test, l_test):
	print('predict')
	res = classifier.predict(i_test)
	def get_precision_score():
		table = list(zip(res,l_test))
		precision_score=1
		for var in table:
			if var[0]==var[1]:
				precision_score+=1

		precision_score/=len(table)

		return precision_score

	print(get_precision_score())

	return res

#-----------------------------------------------------------------------------------
if __name__ == '__main__':

	#load the dataset
	digit_data = datasets.fetch_mldata('MNIST original')

	#rescaling the data between 1, -1
	x = digit_data.data
	x=np.asarray(x, 'float32')
	y = digit_data.target
	'''
	#splitting the dataset into training dataset and testing dataset 
	x = x/255.0*2 - 1
	i_train, i_test, l_train, l_test =cross_validation.train_test_split(x, y, test_size=0.1, random_state=0)
	'''
	#in case we're to expand the data in 4 directions and then manipulate our data
	X, Y = expand_data(x, y)
	X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001) 

	#splitting the dataset into training dataset and testing dataset
	i_train, i_test, l_train, l_test =cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)
	
	def process(i_train, l_train):
		classifier = make_clf(i_train, l_train)
		pickle_data(classifier)

	def evaluate(i_test, l_test):
		'''
		try:
			clf = load_data()
			predict_clf(clf, i_test, l_test)
		except:
			process(i_train, l_train)
			clf = load_data()
			predict_clf(clf, i_test, l_test)
		'''
		
		clf = load_data()
		print(type(clf))
		predict_clf(clf,i_test,l_test)
	evaluate(i_test,l_test)

