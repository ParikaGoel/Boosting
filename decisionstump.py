import numpy as np
from operator import lt,ge

class DecisionStump:
    """
    A simple decision stump classifier
    dim : dimension on which to split
    value : value of dimension
    op : comparator function to use while comparing with the value of the dimesion
    """
    def __init__(self, dim=0, value=0, op=lt):
        self.dim = dim
        self.value = value
        self.op = op
        
    def update(self, dim=None, value=None, op=None):
        if dim is not None: self.dim = dim
        if value is not None: self.value = value
        if op is not None: self.op = op
    
    def predict(self,X):
        return np.array([1 if self.op(x, self.value) else -1 for x in X[:,self.dim]])
    
    """
    Finding an optimal dimension and the value of that dimension which will best split
    the given data depending on which split gives the minimum error
    X : n x d data matrix, n number of samples with d dimension
    Y : n dimensional array containing label of each observation, label = {-1,1}
    sample_weights : weight of each observation
    num_splits : number of split value to be tested randomly
    """
    def fit_data(self,X,Y,sample_weights,num_splits=100):
        n,d = X.shape
        min_err = np.inf
        
        for dim in range(d):
            min_dim_err, value, op = self.fit_dim(X[:,dim],Y,sample_weights,num_splits)
            if min_dim_err < min_err:
                min_err = min_dim_err
                self.update(dim,value,op)
       
    """
    Fit a one dimensional Decision stump classifier
    Finding the optimal value of particular dimension which results in the best split
    if split is performed on that dimension i.e. if I perform on the split on first dimension,
    then what value of the dimension will result in the best split
    This function is called by the fit_data function for every dimension to find the optimal dimension
    """
    def fit_dim(self,X,Y,sample_weights,num_splits):
        min_err, split_value, op = np.inf, None, lt
        num_splits = min(num_splits,len(Y)-1)
        
        for value in np.linspace(min(X),max(X),num_splits,endpoint=False):
            prediction = [1 if x < value else -1 for x in X]
            indicator = np.not_equal(prediction,Y)
            Jm = np.dot(sample_weights,indicator)
            if Jm < min_err:
                min_err = Jm
                split_value = value
                op = lt
            
            Jm = np.dot(sample_weights,~indicator)
            if Jm < min_err:
                min_err = Jm
                split_value = value
                op = ge
         
        return min_err,split_value,op
        