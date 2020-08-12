#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd




df = pd.read_csv('./breastcancer.csv')




df


# In[2]:


X = df.iloc[:,2:-1].values
print(X)
y = df.iloc[:,1].values
print(y)
Yf = y[:]


# In[3]:


# PCA to reduce dataset features to 2 


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

    plt.bar(range(30), indvar, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(30), cumvar, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

wpca = np.hstack((eigenpairs[0][1].reshape(30,1),
                      eigenpairs[1][1].reshape(30,1)))

print('Matrix W:\n', wpca)







 


# In[4]:


Y = stdX.dot(wpca)
print(Y)
b = False
r = False
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    xs = Y[:,0]
    ys = Y[:,1]
    print(xs)
    print(ys)
    for i in range(len(y)):
        xarr = []
        xarr.append(xs[i])
        yarr = []
        yarr.append(ys[i])
        lab = y[i]
        if lab == 'M':
            col = 'blue'
            
            plt.scatter(xarr,
                    yarr,
                    label=lab if b == False else '' ,
                    c=col)
            b = True
        elif lab == 'B':
            col = 'red'
            
            plt.scatter(xarr,
                    yarr,
                    label=lab if r == False else '',
                    c=col)
            r = True
        
        
    
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    
    
    
    
    

    plt.show()
    r = False
    b = False


# In[5]:



# code for regression begins 



Xf = Y[:,:]
#print(y)
Yf = y[:]

for t in range(len(Yf)):
    if Yf[t] == 'B':
        #print("if")
        Yf[t] = 0
    elif Yf[t] == 'M':
        #print("elif")
        Yf[t] = 1
 
#yy = 1 if Yf == 'M'
print(Yf)
#print(yy)
split = int(0.8*Xf.shape[0])
X_train = Xf[:split,:]
Y_train = Yf[:split]
X_test = Xf[split:,:]
Y_test = Yf[split:]


# In[6]:



def hypothesis(x,w,b):
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

def get_acc(x_tst,y_tst,w,b):
    
    y_pred = []
    
    for i in range(y_tst.shape[0]):
        p = predict(x_tst[i],w,b)
        y_pred.append(p)
        
    y_pred = np.array(y_pred)
    
    return  float((y_pred==y_tst).sum())/y_tst.shape[0]


# In[7]:


loss = []
acc = []

W = 2*np.random.random((X_train.shape[1],))
b = 5*np.random.random()


# In[8]:


for i in range(1000):
    l,W,b = gradient_descent(X_train,Y_train,W,b,learning_rate=0.1)
    acc.append(get_acc(X_test,Y_test,W,b))
    loss.append(l)


# In[9]:


plt.plot(loss)
plt.ylabel("Negative of Log Likelihood")
plt.xlabel("Time")
plt.show()


# In[10]:


# Acccuracy with iterations

plt.plot(acc)
plt.show()
print(acc[-1])


# In[11]:


r = False
b = False
plt.figure(0)
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
        elif lab == 0:
            col = 'red'
            
            plt.scatter(xarr,
                    yarr,
                    label=lab if r == False else '',
                    c=col)
            r = True
        
        
    
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
   # plt.legend(loc='lower center')
    plt.tight_layout()
    
    #plt.show()
    r = False
    b = False


x = np.linspace(-4,8,10)
y = -(W[0]*x + b)/W[1]
plt.plot(x,y,color='k')

plt.legend()
plt.show()

