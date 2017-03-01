# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:33:25 2016

@author: Macbook
"""
#import tools 
from tools import *
import numpy as np
from numpy import random

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



class OptimFunc:
    def __init__(self,f=None,grad_f=None,dim=2):
        self.f=f
        self.grad_f=grad_f
        self.dim=dim
                                
    def init(self,low=-1,high=1):
        return random.random(self.dim)*(high-low)+low

def lin_f(x): 
    return x
def lin_grad(x): 
    return 1 

lin_optim=OptimFunc(lin_f,lin_grad,1)         
#Utiliser la fonction : 
lin_optim.f(3)
#le gradient : 
lin_optim.grad_f(1)
				#descente de gradient sur fonction simple	
                        #definition des fonctions
#premiere fonction
def xcosx(x):
    return x*np.cos(x)
def grad_xcosx(x):
    return np.cos(x)-x*np.sin(x)

xcosx_optim=OptimFunc(xcosx,grad_xcosx,1)
#deuxieme fonction
def rosen(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
def grad_rosen(x):
    return [-2*x[0]*200*(x[1]-x[0]**2)+2*x[0]-2 , 200*(x[1]-x[0]**2)]
	
rosen_optim=OptimFunc(rosen,grad_rosen,2)

                        #tracé des fonctions
#xrange=np.arange(-5,5,0.1)
#plt.plot(xrange,xcosx_optim.f(xrange))
#plt.show()

### affichage 3D
#
#xvec=np.arange(-5,5,0.1)
#yvec=np.arange(-5,5,0.1)
#z=rosen_optim.f([xvec,yvec])
#fig = plt.figure()
#ax = fig.gca(projection="3d")
#xvec,yvec= np.meshgrid(xvec,yvec)
#surf = ax.plot_surface(xvec, yvec, z, rstride=1, cstride=1, cmap=cm.gist_rainbow,linewidth=0)
#fig.colorbar(surf)
#plt.show()

                        #descente du gradient
class GradientDescent:
	def __init__(self,optim_f,eps=1e-4,max_iter=5000):
		self.optim_f=optim_f
		self.eps=eps
		self.max_iter=max_iter
		
#		self.i=0
#		self.w = self.optim_f.init()
#		self.log_w=np.array(self.w)
#		self.log_f=np.array(self.optim_f.f(self.w))
#		self.log_grad=np.array(self.optim_f.grad_f(self.w))
								
	def reset(self): 
		self.i=0
		self.w = self.optim_f.init() #point de depart
		self.log_w=np.array(self.w)  #enregistre les points par lesquels on est passé
		self.log_f=np.array(self.optim_f.f(self.w))  #valeurs des fonction au point de depart
		self.log_grad=np.array(self.optim_f.grad_f(self.w)) #valeur du gradient au point de depart
	def optimize(self,reset=True): 
		if reset:
			self.reset()
		while not self.stop():
			self.w = self.w - self.get_eps()*self.optim_f.grad_f(self.w) #new point
			self.log_w=np.vstack((self.log_w,self.w)) 
			self.log_f=np.vstack((self.log_f,self.optim_f.f(self.w))) 
			self.log_grad=np.vstack((self.log_grad,self.optim_f.grad_f(self.w))) 
			if self.i%100==0:
				print self.i," iterations ",self.log_f[self.i] #,self.score(self.data,
			self.i+=1
	def stop(self):
		return (self.i>2) and (self.max_iter and (self.i>self.max_iter)) 
	def get_eps(self): 
		return np.array([self.eps])

						##optimisation de xcosx
#xcos=GradientDescent(xcosx_optim,2e-4)
#xcos.optimize()
##print xcos.log_f
#
#X=np.arange(0,len(xcos.log_f),1)
#plt.figure()
#plt.plot(X,xcos.log_f)
#plt.show()
#
#
#plusieurs figures pour xcos(x)
#eps=1e-4
#for i in range(5):
#	xcos2=GradientDescent(xcosx_optim,eps)
#	xcos2.optimize()
#	Y=xcos2.log_f
#	plt.figure()
#	plt.plot(X,Y)
#	eps=eps+0.001
#plt.show()
#
##trajectoire
#for i in range(5):
#	xcos2=GradientDescent(xcosx_optim,eps)
#	xcos2.optimize()
#	Y=xcos2.log_w
#	plt.figure()
#	plt.plot(X,Y)
#	eps=eps+0.001
#plt.show()


#						#optimisation de rosen
#rosen_descent=GradientDescent(rosen_optim)
#rosen_descent.optimize()
#
##tracé descente en valeur
#eps=0.01
#for i in range(5):
#	rosen_descent2=GradientDescent(rosen_optim)
#	rosen_descent2.optimize()
#	X=np.arange(0,len(rosen_descent2.log_f),1)
#	Y=rosen_descent2.log_f
#	plt.figure()
#	plt.plot(X,Y)
#	eps=eps+0.01
#plt.show()
#
##tracé trajectoire
#
#X=rosen_descent.log_w[0]
#Y=rosen_descent.log_w[1]
#plt.figure()
#plt.plot(X,Y)
#plt.show()

### affichage 3D
##
#mpl.rcParams['legend.fontsize'] = 10
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#x=rosen_descent.log_w[0]
#y=rosen_descent.log_w[1]
#z=rosen_descent.log_f
#ax.plot(x, y, z, label='Trajectoire Rosen')
#ax.legend()
#
#plt.show()

						#régression linéaire
def gen_1d(n,eps=0.01):
	X=10*np.random.random(n)-5
	Y=2*X+1+np.random.normal(0,eps**2,n)
	return X,Y
	
a=gen_1d(10)
print len(a[1]), len(a[0])

#plot de la fonction bruitée
test=gen_1d(100)
plt.figure()
plt.plot(test[0],test[1])
plt.show()

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
				
#fonction de coût 
def hinge_loss(X,Y,w):
#	to_line(X)
#	to_col(w)
#	to_col(Y)
	n=Y.shape[0]
	M=X.transpose().dot(w)-Y
	return M.dot(M)/(2*n)

def hinge_grad(X,Y,w):
	to_col(w)
	to_col(Y)
	to_line(X)
	n=Y.shape[0]
	M=X.dot( X.transpose() )
	return -1/n*(M.dot(w) - X.dot(Y))
	
class Regression(Classifier,GradientDescent,OptimFunc): 
	def __init__(self,eps=1e-4,max_iter=5000):
		GradientDescent.__init__(self,self,eps,max_iter)
		self.dim=self.data=self.y=self.n=self.w=None 
		
	def fit(self,data,y):
		self.y=y
		self.n=y.shape[0]              
		self.dim=data.size/self.n+1
		self.data=data.reshape((self.n,self.dim-1))
		self.data=np.hstack( ( np.ones( (self.n,1) ), self.data))
		print 'ajout des 1', self.data.shape[0]
		self.optimize()
		
	def f(self,w):
		#On met la matrice des données en ligne
		if self.data.shape[0]>self.data.shape[1]:
			self.data=self.data.transpose()
#		w=to_col(w)
#		self.y=to_col(self.y)
#		self.data=to_line(self.data)
		print 'nombre de lignes de data:',self.data.shape[0]
		M=self.data.transpose().dot(w)-self.y
		print 'vecteur w',w,'taille de M',M.shape[0],'echantillon:',self.n
		print 'coût :',M.transpose().dot(M)/(2*self.n)
		return M.transpose().dot(M)/(2*self.n)
		
	def grad_f(self,w): 
		if self.data.shape[0]>self.data.shape[1]:
			self.data=self.data.transpose()
#		w=to_col(w)
#		self.y=to_col(self.y)
#		self.data=to_line(self.data)
		M=self.data.dot( self.data.transpose() )
#		print 'matrice M',M,'vecteur w',w
#		print ' on utilise le gradient',1/self.n*( M.dot(w) - self.data.dot(self.y) )
		return ( M.dot(w) - self.data.dot(self.y) )/self.n
	def init(self):
		return np.random.random(self.dim)*(np.max(self.data)-np.min(self.data))+np.min(self.data)
	def predict(self,data):
		n=data.size/(self.dim-1)
		return np.hstack((np.ones((n,1)),data.reshape(n,self.dim-1))).dot(self.w)


data,y=gen_1d(50)
reg=Regression(0.05,500)
reg.fit(data,y)
print reg.w
#np.dot()
