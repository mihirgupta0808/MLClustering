#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


df = pd.read_csv('./Winerandom.csv')
df.sample(frac=1)
print(df)
print(df.iloc[0].values)
df.tail()


# In[2]:


print(df)


# In[3]:


import random
X = df.iloc[:,1:].values
y = df.iloc[:,0].values
print(X)
print(y)

'''
temp = list(zip(X, y)) 
random.shuffle(temp) 
res1, res2 = zip(*temp) 
X = list(res1)
y = list(res2)
print(res1)
print(res2)
'''


# In[4]:



# PCA

from sklearn.preprocessing import StandardScaler
stdX = StandardScaler().fit_transform(X)


import numpy as np
means = np.mean(stdX, axis=0)
covariances = (stdX - means).T.dot((stdX - means)) / (stdX.shape[0]-1)
print('Covariance matrix \n%s' %covariances)

print('NumPy covariance matrix: \n%s' %np.cov(stdX.T))

covariances = np.cov(stdX.T)

eigenvals, eigenvecs = np.linalg.eig(covariances)

print('Eigenvectors \n%s' %eigenvecs)
print('\nEigenvalues \n%s' %eigenvals)

cmat = np.corrcoef(X.T)

eigenvals, eigenvecs = np.linalg.eig(cmat)

print('Eigenvectors \n%s' %eigenvecs)
print('\nEigenvalues \n%s' %eigenvals)


#  (eigenvalue, eigenvector) 
eigenpairs = [(np.abs(eigenvals[i]), eigenvecs[:,i]) for i in range(len(eigenvals))]

# Sort the eignepairs in decreeaseing order of eigenvalue
eigenpairs.sort(key=lambda x: x[0], reverse=True)


print('Eigenvalues in descending order:')
for x in eigenpairs:
    print(x[0])

from matplotlib import pyplot as plt
total = sum(eigenvals)
indvar = [(x / total)*100 for x in sorted(eigenvals, reverse=True)]
cumvar = np.cumsum(indvar)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(13), indvar, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(13), cumvar, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

wpca = np.hstack((eigenpairs[0][1].reshape(13,1),
                      eigenpairs[1][1].reshape(13,1)))

print('Matrix W:\n', wpca)


# In[5]:


Y = stdX.dot(wpca)
#       Y contains features and y contains labels 
print(Y)
print(y)
split = int(0.8*len(y))
X_train = Y[:split,:]
Y_train = y[:split]
X_test = Y[split:,:]
Y_test = y[split:]

plt.figure(0)
r = False
b = False
g = False

print(len(X_train))
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    #xs = Y[:,0]
    #ys = Y[:,1]
    #print(xs)
    #print(ys)
    for i in range(len(X_train)):
        xarr = []
        xarr.append(X_train[i,0])
        yarr = []
        yarr.append(X_train[i,1])
        lab = Y_train[i]
        #print(lab)
        if lab == 1:
            col = 'blue'
            
            plt.scatter(xarr,
                    yarr,
                    label=lab if b == False else '' ,
                    c=col)
            b = True
        elif lab == 2:
            col = 'red'
            
            plt.scatter(xarr,
                    yarr,
                    label=lab if r == False else '',
                    c=col)
            r = True
        elif lab == 3:
            col = 'green'
            
            plt.scatter(xarr,
                    yarr,
                    label=lab if g == False else '',
                    c=col)
            g = True
        
        
    
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    

    plt.show()
    r = False
    b = False
    g = False


# In[6]:



#  Multinomial Regression

def hypothesis(x,w,b):
    '''accepts input vector x, input weight vector w and bias b'''
    hx = np.dot(x,w)+b
    return sigmoid(hx)
def sigmoid(h):
    return 1.0/(1.0 + np.exp(-1.0*h))
def error(y,x,w,b):
    m = x.shape[0]
    err = 0.0
    for i in range(m):
        hx = hypothesis(x[i],w,b)
        err += y[i]*np.log2(hx)+(1-y[i])*np.log2(1-hx)
    return err/m
def get_grad(x,w,b,y):
    grad_b = 0.0
    grad_w = np.zeros(w.shape)
    m = x.shape[0]
    for i in range(m):
        hx = hypothesis(x[i],w,b)
        grad_w += (y[i] - hx)*x[i]
        grad_b +=  (y[i]-hx)
    
    grad_w /=m
    grad_b /=m
    return [grad_w,grad_b]
def gradient_descent(x,y,w,b,learning_rate=0.01):
    err = error(y,x,w,b)
    [grad_w,grad_b] = get_grad(x,w,b,y)
    w = w + learning_rate*grad_w
    b = b + learning_rate*grad_b
    return err,w,b
def predict(x,w,b):
    
    confidence = hypothesis(x,w,b)
    if confidence<0.5:
        return 0
    else:
        return 1
    
def get_prob(x,w,b):
    confidence = hypothesis(x,w,b)
    return confidence
    
def get_acc(x_tst,y_tst,w,b):
    
    y_pred = []
    
    for i in range(y_tst.shape[0]):
        p = predict(x_tst[i],w,b)
        y_pred.append(p)
        
    y_pred = np.array(y_pred)
    
    return  float((y_pred==y_tst).sum())/y_tst.shape[0]


# In[7]:


print(Y_train)
print(len(X_train))
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for i in range(len(X_train)):
        xarr = []
        xarr.append(X_train[i,0])
        yarr = []
        yarr.append(X_train[i,1])
        lab = Y_train[i]
        #print(lab)
        if lab == 1:
            col = 'blue'
            
            plt.scatter(xarr,
                    yarr,
                    label=lab if b == False else '' ,
                    c=col)
            b = True
        elif lab == 2:
            col = 'red'
            
            plt.scatter(xarr,
                    yarr,
                    label=lab if r == False else '',
                    c=col)
            r = True
        elif lab == 3:
            col = 'green'
            
            plt.scatter(xarr,
                    yarr,
                    label=lab if g == False else '',
                    c=col)
            g = True
        
        
    
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
   # plt.legend(loc='lower center')
    plt.tight_layout()
    #plt.show()
    r = False
    b = False
    g = False
probs = []
weights = []
biases = []
for lab in [1,2,3]:
    print("label is " , lab)
    
    
    Xn = X_train[:,:]
    
    #print(y)
    Yf = Y_train[:]
    
    print(Yf)
    Yn = []
    print(Yn)
    for t in range(len(Y_train)):
        
       
        if Yf[t] == lab:
            #print("if")
            Yn.append(1)
        
        else :
            #print("elif")
            Yn.append(0)
        
   # Xf = Y[:,:]
    #Yf = y[:]
    print(Yn)
    # yn is no wthe yf for prediction !!
    # Xn and Yn are used for training 
    loss = []
    acc = []

    W = 2*np.random.random((Xn.shape[1],))
    b = 5*np.random.random()
    for i in range(1000):
        l,W,b = gradient_descent(Xn,Yn,W,b,learning_rate=0.1)
        weights.append(W)
        biases.append(b)
        acc.append(get_acc(X_test,Y_test,W,b))
        loss.append(l)
    prob = []
    
    for i in range(Y_test.shape[0]):
        p = get_prob(X_test[i],W,b)
        prob.append(p)
        
    prob = np.array(prob)
    probs.append(prob)
    

    #plt.scatter(dist_01[:,0],dist_01[:,1],label='Class 0')
    #plt.scatter(dist_02[:,0],dist_02[:,1],color='r',marker='^',label='Class 1')
    #plt.xlim(-5,10)
    #plt.ylim(-5,10)
    #plt.xlabel('x1')
    #plt.ylabel('x2')
    
    
   
    
    
    
    plt.scatter(list(X_test[:,0]), list(X_test[:,1]),label='test',c='yellow') 
    
    
    
    x = np.linspace(-4,8,10)
    y = -(W[0]*x + b)/W[1]
    plt.plot(x,y,color='k')
    
    
    
plt.legend()
plt.show()


# In[8]:


# work with weights,biases
# calculate probs then find max 
print(probs)
#pmax = np.maximum(prob[0],prob[1],prob[2])
 #np.maximum([2, 3, 4], [1, 5, 2])
#probs
s = probs[0].size
print(s)
plabs = []
for i in range(s):
    if probs[0][i] > probs[1][i] :
        clab = 1
        tempp = probs[0][i]
    else :
        clab = 2
        tempp = probs[1][i]
    if tempp > probs[2][i] : 
        plab = tempp
        
    else:
        plab = probs[2][i]
        clab = 3
    plabs.append(clab)
    
    
plabs =  np.array(plabs)
    
    
print(plabs,type(plabs))


print(Y_test,type(Y_test))

    

        
accuracy = (float((plabs==Y_test).sum())/s)*100  
print("accuracy is ", accuracy , " %")
    
    


# In[9]:


accuracy = (float((plabs==Y_test).sum())/s)*100  
print("accuracy is ", accuracy , " %")


# In[10]:


plt.scatter(list(X_test[:,0]), list(X_test[:,1]),label='test',c='yellow') 


# In[11]:


print(X_train)
plt.scatter(X_train[:,0],X_train[:,1])


# In[12]:



# meanshift clustering 




import matplotlib.pyplot as plt
import numpy as np


class Find_Centroids:
    def __init__(self, rad=2.5):
        self.rad = rad

    def fit(self, Xvec):
        centroids = {}

        for i in range(len(Xvec)):
            centroids[i] = Xvec[i]
        
        while True:
            newcentroids = []
            for i in centroids:
                nearby = []
                centroid = centroids[i]
                for example in Xvec:
                    if np.linalg.norm(example-centroid) < self.rad:
                        nearby.append(example)

                newcentroid = np.average(nearby,axis=0)
                newcentroids.append(tuple(newcentroid))

            uniques = sorted(list(set(newcentroids)))

            oldcentroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            converge = True

            for i in centroids:
                if not np.array_equal(centroids[i], oldcentroids[i]):
                    converge = False
                if not converge:
                    break
                
            if converge:
                break

        self.centroids = centroids



obj = Find_Centroids()
obj.fit(X_train[:,:])

centroids = obj.centroids

plt.scatter(X_train[:,0], X_train[:,1])

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='b', marker='*')

plt.show()
print(centroids)
print(type(centroids))


# In[13]:


k = 5

color = ['green', 'red', 'blue', 'yellow', 'gray','orange','purple']
clusters = {}

for i in range(k):
    #center = 10*(2*np.random.random((X.shape[1],))-1)
    center = centroids[i]
    points = []
    
    cluster = {
        'center':center, 
        'points':points,
        'color':color[i]
    }
    
    clusters[i] = cluster
def distance(v1,v2):
    return np.sqrt(np.sum((v1-v2)**2))
def assignPointToClusters(clusters): #E-step 

    for ix in range(X_train.shape[0]):
        
        dist = []
        for kx in range(k):
            d = distance(X_train[ix], clusters[kx]['center'])
            dist.append(d)
            
        current_cluster = np.argmin(dist)
        clusters[current_cluster]['points'].append(X_train[ix])
def updateClusters(clusters): #M-Step -> We update every cluster center according to the mean of the points
    for kx in range(k):
        pts = np.array(clusters[kx]['points'])
        
        if pts.shape[0]>0: # if cluster has some non-zero points
            new_mean = pts.mean(axis=0)
            clusters[kx]['center'] = new_mean
            #clear my points list 
            clusters[kx]['points'] = [] 
def plotClusters(clusters):
    
    for kx in range(k):
        
        pts = np.array(clusters[kx]['points'])
        
        try:
            plt.scatter(pts[:,0], pts[:,1], color=clusters[kx]['color'])
        except:
            pass
        
        # plot the cluster center
        uk = clusters[kx]['center']
        plt.scatter(uk[0], uk[1], color='black', marker='*')


assignPointToClusters(clusters)
plotClusters(clusters)
        


# In[14]:


count =1 
while True:
    assignPointToClusters(clusters)
    updateClusters(clusters)
    count+=1
    if count>1000:
        break
assignPointToClusters(clusters)
plotClusters(clusters)
updateClusters(clusters)


# In[ ]:




