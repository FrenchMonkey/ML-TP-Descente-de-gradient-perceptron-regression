# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 08:56:52 2016

@author: Macbook
"""

import numpy as np
from numpy import random
from tools import *
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
	def __init__(self,max_iter = 100,eps=8e-4):
		self.max_iter = max_iter
		self.eps = eps
	def fit(self,data,y):
		self.w = np.random.random((1, data.shape[1])) 
		self.hist_w = np.zeros((self.max_iter, data.shape[1])) 
		self.hist_f = np.zeros((self.max_iter, 1))
		self.i = 0
		while self.i < self.max_iter :
			self.w = self.w + self.eps * hinge_grad(self.w, data, y) ## A completer 
			self.hist_w[self.i] = self.w
			self.hist_f[self.i] = hinge(self.w, data,y)
			#if self.i % 100==0: print self.i,self.hist_f[self.i]
			self.i += 1
	
	def predict(self,data): ## A completer
		self.w=to_col(self.w)
		fw=data.dot(self.w.T)
		self.resultat_predit=np.sign(fw)
		return np.sign(fw)
		
	def score(self,data,y): ## A completer
		prediction=self.predict(data)
		y=to_col(y)
#		print 'shape y',np.shape(y),'shape predict',np.shape(prediction)
		m=(( prediction-y)==0 ).astype(int)	
		return np.mean(m)

#test
#test=Perceptron()
#test.fit(data,y)
#test.predict(data)
#print "score:", test.score(data,y)
	

### Generer et tracer des donnees
	#test 1 type 
#
#datax,datay = gen_arti(data_type=0,nbex=1000,epsilon=0.1) 
##print datay.shape, datax.shape
#
#plt.figure()
#
##datay = datay.reshape(datay.shape[0],1)
#
#p=Perceptron()
#p.fit(datax,datay)
#projection=p.predict(datax)
#print projection-datay.reshape(datay.shape[0],1)
#print 'score du fitting type 0', p.score(datax,datay) 
##print 'vecteur w', p.w
#plot_frontiere(datax,p.predict,50) 
#plot_data(datax,datay)
#
##print p.predict(datax)
#
#	#test 2 type 1
#plt.figure()
#datax,datay = gen_arti(data_type=1,nbex=1000,epsilon=0.5) 
#p=Perceptron()
#p.fit(datax,datay)
#print 'score du fitting type1', p.score(datax,datay) 
#plot_frontiere(datax,p.predict,50) 
#plot_data(datax,datay)
#
#	
#	#test 2 type 2
#	
#plt.figure()
#datax,datay = gen_arti(data_type=2,nbex=1000,epsilon=0.5) 
#p=Perceptron()
#p.fit(datax,datay)
#print 'score du fitting type2', p.score(datax,datay) 
#plot_frontiere(datax,p.predict,50) 
#plot_data(datax,datay)


						#implementation du poids w0
#datax_2=np.hstack( ( np.ones( (datax.shape[0],1) ), datax) )
#
#p2=Perceptron()
#p2.fit(datax_2,datay)
#
#fw=datax_2.dot(p2.w.T)
#resultat_predit=np.sign(fw)
#masque=(resultat_predit-datay)==0
#score2=np.mean(masque)
#
##print 'w',p2.w,'data',datax_2
#print 'score du fitting avec w0', p2.score(datax_2,datay) 
#print 'vecteur w', p2.w


#version stochastique du perceptron




# PARTIE 2: DIFFERENCIER DES CHIFFRES MANUSCRITS

def load_usps(filename):
	with open(filename ,"r") as f:
		f.readline()
		data = [[float(x) for x in l.split()] for l in f if len(l.split())>2] 
	tmp = np.array(data)
	return tmp[:,1:],tmp[:,0].astype(int)
	
X_source,y = load_usps('chiffres.txt');
y_prim =  y.reshape(y.shape[0],1);

#on prend une partie des données pour tester l'algo sur le reste
X=X_source[0:6750,0:6750]

#plot_data(X,y);
#
##test 6 vs 9
##
p=Perceptron();
indices_6=(y==6);
indices_9=(y==9);

X_6=X[indices_6,:]
X_9=X[indices_9,:]
print 'taille X_6', X_6.shape[0]
print 'taille X_9', X_9.shape[0]
dataX=np.vstack( (X_6,X_9) )
print 'taille dataX', dataX.shape[0]
#
y_6=np.ones( [X_6.shape[0],1] ) #la classe 1 est la classe 6
y_9=-np.ones( [(X_9.shape[0]),1] )
dataY=np.vstack( (y_6,y_9) )
print 'taille dataY', dataY.shape[0]
#
#on implemente une colonne de 1 pour avoir le biais
#dataX=np.hstack( ( np.ones( [dataX.shape[0],1] ), dataX ) )
#print 'taille dataX colonnes', dataX.shape[1]

#algorithme perceptron
p.fit(dataX,dataY)
print 'score de la classification 6vs9', p.score(dataX,dataY)

x=np.arange(p.max_iter)
plt.figure()
plt.plot(x,p.hist_f)

#test 6 vs all

p=Perceptron();

indices_6=(y==6);
indices_autres=(y!=6);

X_6=X[indices_6,:]
X_autres=X[indices_autres,:]
dataX=np.vstack( (X_6,X_autres) )
#print 'taille dataX', dataX.shape[0]

#on implemente une colonne de 1 pour avoir le biais
#dataX=np.hstack( ( np.ones( [dataX.shape[0],1] ), dataX ) )
#print 'taille dataX colonnes', dataX.shape[1]

y_6=np.ones( [X_6.shape[0],1] ) #la classe 1 est la classe 6
y_autres=-np.ones( [(X_autres.shape[0]),1] )
dataY=np.vstack( (y_6,y_autres) )
#print 'taille dataY', dataY.shape[0]

#algorithme perceptron
p.fit(dataX,dataY)
print 'score de la classification 6vsall', p.score(dataX,dataY)

x=np.arange(p.max_iter)
plt.figure()
plt.plot(x,p.hist_f)

#PARTIE 3: EXTENSIONS LINEAIRES

def proj_poly2(data):
	if (data.shape[0]!=2 and data.shape[1]!=2):
		print 'Trop de variables en entrée'
	if data.shape[0]<data.shape[1]:
		data=data.transpose()
	c_3=data[:,0]*data[:,1]
	c_3=to_col(c_3)
	c_4=data[:,0]*data[:,0]
	c_4=to_col(c_4)
	c_5=data[:,1]*data[:,1]
	c_5=to_col(c_5)
#	print "data", data.shape[0],'c_3',c_3.shape[0],'c_4',c_4.shape[0],'c_5',c_5.shape[0]
#	print "data", data.shape[1],'c_3',c_3.shape[1],'c_4',c_4.shape[1],'c_5',c_5.shape[1]

	return np.hstack( (np.ones([data.shape[0],1]),data,c_3,c_4,c_5) )
	
data,datay = gen_arti(data_type=1,nbex=1000,epsilon=1) 

data2=proj_poly2(data)

#algorithmique
p=Perceptron()
p.fit(data2,datay)
print 'score de classification elliptique', p.score(data2,datay)

x=np.arange(p.max_iter)
plt.figure()
plt.plot(x,p.hist_f)

 
