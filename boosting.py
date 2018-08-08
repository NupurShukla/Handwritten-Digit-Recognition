import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		x=np.array(features)

		f=np.zeros(x.shape[0])
		for t in range(self.T):
			decstump=DecisionStump(self.clfs_picked[t].s,self.clfs_picked[t].b,self.clfs_picked[t].d)
			f=f+(self.betas[t]*np.array(decstump.predict(features)))

		predictions=np.ones(f.shape, np.int)
		predictions[np.where(f<0)[0]]=-1
	
		return predictions.tolist()


class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		X=np.array(features)
		N=X.shape[0]
		w=np.full((N),1/N)
		labels=np.array(labels)
		hx=np.array((N))

		for t in range(self.T):

			#step3
			min=9223372036854775807
			for clf in self.clfs:
				decstump=DecisionStump(clf.s,clf.b,clf.d)
				
				hxpred=np.array(decstump.predict(features))
				indicator=np.zeros((N))
				indicator[np.where(labels!=hxpred)[0]]=1

				check=np.sum(np.multiply(w,indicator))

				if check < min:
					min_clf=clf
					hx=hxpred
					min=check

			self.clfs_picked.append(min_clf)
			
			error=0
			for i in range(N):
				if labels[i]!=hx[i]:
					error=error+w[i]

			beta=(1/2)*np.log((1-error)/error)
			self.betas.append(beta)
			for i in range(N):
				if labels[i]==hx[i]:
					w[i]=w[i]*np.exp((-1)*self.betas[t])
				else:
					w[i]=w[i]*np.exp(self.betas[t])

			w_sum=np.sum(w)
			w=w/w_sum

		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		X=np.array(features)
		N=X.shape[0]
		pi=np.full((N),1/2)
		y=np.array(labels)
		f=np.zeros((N))
		hx=np.array((N))

		for t in range(self.T):
			num=((y+1)/2)-pi
			den=np.multiply(pi,1-pi)
			z=num/den
			w=np.multiply(pi, 1-pi)

			#step5
			min=9223372036854775807
			for clf in self.clfs:
				decstump=DecisionStump(clf.s,clf.b,clf.d)
				hxpred=np.array(decstump.predict(features))
				check=np.sum(np.multiply(w,np.multiply(z-hxpred,z-hxpred)))
				
				if check < min:
					min_clf=clf
					hx=hxpred
					min=check

			self.clfs_picked.append(min_clf)
			self.betas.append(0.5)

			f=f+(1/2)*hx
			den=1+np.exp(-2*f)
			pi=1/den


	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	