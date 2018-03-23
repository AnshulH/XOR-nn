import numpy as np

Inp=np.array([[0,0],[0,1],[1,0],[1,1]])
Out=np.array([[0,1,1,0]]).T

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_der(z):
    return z*(1-z)

def forwardProp(Inp,hl1,hl2):
    Inp = np.c_[np.ones(Inp.shape[0]),Inp]
    L1 = np.dot(Inp,hl1)
    A1 = sigmoid(L1)
    A1 = np.c_[np.ones(A1.shape[0]),A1]
    L2 = np.dot(A1,hl2)
    A2 = sigmoid(L2)
    return L1,A1,L2,A2

def backProp(Inp,Out,hl1,hl2,iter):
    for i in range(iter):
        L1, A1, L2, A2 = forwardProp(Inp,hl1,hl2)
        
        ErrorOut = Out - A2
        ErrorL2 = ErrorOut * sigmoid_der(A2)
                
        ErrorL1 = ErrorOut.dot(hl2[1:,:].T)*sigmoid(np.c_[np.ones(Inp.shape[0]),Inp].dot(hl1))
        delL1 = ErrorL1 * sigmoid_der(A1[:,1:])

        hl2 += Alpha*A1.T.dot(ErrorL2)
        hl1 += Alpha*np.c_[np.ones(Inp.shape[0]),Inp].T.dot(delL1)
        
    return A2

np.random.seed(1)
hl1 = np.random.random((3, 2))
hl2 = np.random.random((3, 1))     
iter = 5000
Alpha = .5
print(backProp(Inp,Out,hl1,hl2,iter))