# -*- coding: utf-8 -*-
from numpy import linalg as la
import numpy as np

def myCov(a):
    at = a.T
    [rows,cols]=np.shape(at);
#    print("rows=", rows, "cols=", cols)
    rv = np.ndarray(shape=(rows,rows), dtype=float, order='F')
    for i in range(rows):
        for j in range(rows):
            rv[i][j] = (np.dot(at[i]-np.mean(at[i]), at[j]-np.mean(at[j])))/cols
    return rv

def pro1(A):
    print("\n\nproblem1:\n")
    c1 = np.cov(A.T, None, True, True);
    c2 = myCov(A);
    print("c1=\n", c1)
    print("c2=\n", c2)
    print("c1=c2", abs(c1-c2) < 0.00001)
    
def pro2(A):
    print("\n\nproblem2:\n")
    B = myCov(A)
    eigvalues, eigvectors = la.eig(B)
        
    eigvectors12 = eigvectors[:, 0:2]
    eigvec1 = eigvectors12[:,0]
    eigvec2 = eigvectors12[:,1]
    
    print("eigvec1=", eigvec1)
    print("eigvec2=", eigvec2)
    projA = np.matmul(A , eigvectors12)
    
    print("reduced data=", projA)
      
    covProjA = myCov(projA)
    
    print("covariance matrix=", covProjA)
    varProjA = np.sum(covProjA)
    varProjA = np.matmul(np.matmul(eigvec1, B), eigvec1.T) + np.matmul(np.matmul(eigvec2, B), eigvec2.T)
   
    print("variance=", varProjA)
    print("variance=eigv1+eigv2", abs(varProjA - eigvalues[0] - eigvalues[1]) < 0.00001)
    
    print ("eigvals=", eigvalues)
    print ("eigvectors=", eigvectors)
    
def pro3(A):
    print("\n\nproblem3:\n")
    B = myCov(A)
    [rows,cols]=np.shape(A)
    #print("rows=", rows, "cols=", cols)   
    eigvalues, eigvectors = la.eig(B)
    eigvectors12 = eigvectors[:, 0:2]
    
        
    print("eigvec12=", eigvectors12)
    print("eigvec12t=", eigvectors12.T)
    P2=np.matmul(eigvectors12, eigvectors12.T)
    #P2 = np.matmul( eigvec1.T, eigvec1) + np.matmul(eigvec2.T, eigvec2)
    print("projection matrix=", P2)
    
    #projA=np.matmul(A, P2)
    
    #projA=np.matmul(A, P2)
    projA1 = np.matmul(P2, A.T)
    projA = projA1.T
    print("projected datapoints=", projA) 
    MSE = np.trace(B) - np.trace(myCov(projA))
    print("MSE=", MSE)
    
    eigvalsum8 = 0
    for i in range(2, 10):
        eigvalsum8 += eigvalues[i]
    print ("sum of 8 eigen vaulues=", eigvalsum8)
    print("Error=Sum of 8 eigen values", abs(eigvalsum8 - MSE) < 0.0001)
    #print ("eigvals=", eigvalues)    
    #print ("eigvectors=", eigvectors)
        
def pro4(A):
     print("\n\nproblem4:\n")
     B = myCov(A)
     eigvalues, eigvectors = la.eig(B)
     print("Uðﾝﾚﾲðﾝﾐﾔ^ðﾝﾑﾇ=", eigvectors, "*",np.diag(eigvalues), "*", eigvectors.T)

def pca(A, alpha):
    B = myCov(A)
    [rows,cols]=np.shape(A)
    eigvalues, eigvectors = la.eig(B)
    eigvalsum = np.sum(eigvalues)
    
    for r in range(0, cols):
        eigpartsum = np.sum(eigvalues[0:r+1])
        #print("sum=", eigvalsum, "part sum=", eigpartsum)
        if eigpartsum / eigvalsum >= alpha:
            #print("r=", r)
            break
    if r == cols:
        return None
    reducedA = np.matmul(A, eigvectors[:, 0:r+1])
    return {"dim":r, "matrix":reducedA}
def pro5(A):
    print("\n\nproblem5:\n")
    reducedA = pca(A, 0.5)
    print("reducedA with alpha=0.5 ", reducedA["matrix"])    
    
    reducedA = pca(A, 0.8)
    print("reducedA with alpha=0.8 ", reducedA["matrix"])    
def pro6(A):
    print("\n\nproblem6:\n")
    reducedA = pca(A, 0.9)
    print("co-ordinate of 10 data points as new basis=", reducedA["matrix"][:10,:])
    
def main():
    float_formatter = lambda x: "%.4f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    A = np.loadtxt('magic04.txt', delimiter=',', usecols=range(10))
    print ("input matrix=",A)
    pro1(A)
    pro2(A)
    pro3(A)
    pro4(A)
    pro5(A)
    pro6(A)
#print("start")
main()

    
