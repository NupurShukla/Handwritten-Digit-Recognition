import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
	def __init__(self, s:int, b:float, d:int):
		self.clf_name = "Decision_stump"
		self.s = s
		self.b = b
		self.d = d

	def train(self, features: List[List[float]], labels: List[int]):
		pass
		
	def predict(self, features: List[List[float]]) -> List[int]:
		features=np.array(features)
		xds=features[:,self.d]

		prediction=np.zeros(xds.shape)
		prediction[np.where(xds>self.b)[0]]=self.s
		prediction[np.where(xds<=self.b)[0]]=(-1)*self.s
		return prediction.tolist()
		