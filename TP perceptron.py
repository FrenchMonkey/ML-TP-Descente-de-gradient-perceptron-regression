# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 08:56:52 2016

@author: Macbook
"""

import numpy as np
from numpy import random
import tools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


def to_col(x):
    """ convert an vector to column vector if needed """
    if len(x.shape)==1:
        x=x.reshape(x.shape[0],1)
    return x
def to_line(x):
    """ convert an vector to line vector if needed """
    if len(x.shape)==1:
        x=x.reshape(1,x.shape[0])
    return x

w = np.random.random((1,3))
data = np.random.random((5,3))
y = np.random.randint(0,2,size = (5,1))*2-1
		
#fonction hinge
def hinge(w,data,y,alpha=0):
	#passe en vecteur la matrice au cas où  n==1
	data=to_line(data)
	w=to_col(w)
	y=to_col(y)
	#vecteur fw
	fw=data.dot(w.T)
	vec=y*fw
	vec=np.maximum(0,vec)
	return vec.mean()

def hinge_grad(w,data,y,alpha=0):
	#passe en vecteur la matrice au cas où  n==1
	data=to_line(data)
	w=to_line(w)
	#transforme data en sommant les lignes
	one=np.ones(w.size)
	X=data.dot(one)
	#multiplie par y par composante
	vec=X*y
	#inclus le signe
	#vecteur fw
	fw=data.dot(w.T)
	#enleve les signes positifs
	vec_2=(fw*y)<0
	#le vecteur gradient
	grad=vec*vec_2
	return np.mean(grad)
	
#tests fonction
	
##### doit retourner un scalaire
#print hinge(w,data,y), hinge(w,data[0],y[0]), hinge(w,data[0,:],y[0])
#### doit retourner un vecteur de taille (w.shape[1],)
#print hinge_grad(w,data,y),hinge_grad(w,data[0],y[0]),hinge_grad(w,data[0,:],y[0])
	

class Perceptron:
	def __init__(self,max_iter = 1000,eps=1e-3):
		self.max_iter = max_iter
		self.eps = eps
	def fit(self,data,y):
		self.w = np.random.random((1,data.shape[1])) 
		self.hist_w = np.zeros((self.max_iter,data.shape[1])) 
		self.hist_f = np.zeros((self.max_iter,1))
		self.i=0
		while self.i < self.max_iter :
			self.w = self.w + self.eps*hinge_grad(w,data,y) ## A completer 
			self.hist_w[self.i]=self.w 
			self.hist_f[self.i]=hinge(self.w,data,y)
			#if self.i % 100==0: print self.i,self.hist_f[self.i]
			self.i+=1
	
	def predict(self,data): ## A completer
		fw=data.dot(self.w.T)
		self.resultat_predit=np.sign(fw)
		return np.sign(fw)
	def score(self,data,y): ## A completer
		masque=(self.resultat_predit-y)==0
		return np.mean(masque)

#test
test=Perceptron()
test.fit(data,y)
test.predict(data)
print "score:", test.score(data,y)
	
	
