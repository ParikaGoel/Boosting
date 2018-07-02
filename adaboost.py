import numpy as np
from decisionstump import DecisionStump

class Adaboost:
    def __init__(self, weak_learners=None, learner_weights=None):
        self.weak_learners = weak_learners if weak_learners is not None else []
        self.learner_weights = learner_weights if learner_weights is not None else []
    
    def train(self, X, Y, num_classifier, weak_learner_class = DecisionStump):
        n,d = X.shape
        sample_weights = np.repeat(1/n,n);
        
        for m in range(num_classifier):
            weak_learner = weak_learner_class()
            weak_learner.fit_data(X,Y,sample_weights)
            learner_prediction = weak_learner.predict(X)
            learner_indicator = np.not_equal(learner_prediction,Y)
            learner_error = np.dot(sample_weights,learner_indicator) / np.sum(sample_weights)
            learner_error = max(1e-8,learner_error) #Avoid division by zero
            learner_weight = np.log((1-learner_error)/learner_error)
            self.weak_learners.append(weak_learner)
            self.learner_weights.append(learner_weight)
            sample_weights = [sw * np.exp(learner_weight*im) for sw,im in zip(sample_weights,learner_indicator)]
       
    def add_learner(self, X, Y, weak_learner_class = DecisionStump):
        n,d = X.shape
        sample_weights = np.repeat(1/n,n);
        
        for l,w in zip(self.weak_learners,self.learner_weights):
            learner_prediction = l.predict(X)
            learner_indicator = np.not_equal(learner_prediction,Y)
            learner_error = np.dot(sample_weights,learner_indicator) / np.sum(sample_weights)
            learner_error = max(1e-8,learner_error) #Avoid division by zero
            learner_weight = np.log((1-learner_error)/learner_error)
            
            #Update data weighting coefficient for next learner
            sample_weights = [sw * np.exp(learner_weight*im) for sw,im in zip(sample_weights,learner_indicator)]
            
        new_weak_learner = weak_learner_class()
        new_weak_learner.fit_data(X,Y,sample_weights)
        new_learner_prediction = new_weak_learner.predict(X)
        new_learner_indicator = np.not_equal(new_learner_prediction,Y)
        new_learner_error = np.dot(sample_weights,new_learner_indicator) / np.sum(sample_weights)
        new_learner_error = max(1e-8,new_learner_error) #Avoid division by zero
        new_learner_weight = np.log((1-new_learner_error)/new_learner_error)
        self.weak_learners.append(new_weak_learner)
        self.learner_weights.append(new_learner_weight)
    
    def predict(self,X):
        n,d = X.shape
        predictions = np.zeros(n)
        for l,w in zip(self.weak_learners,self.learner_weights):
            learner_prediction = l.predict(X)
            predictions += w * learner_prediction
            
        predictions = np.array([1 if p>0 else -1 for p in predictions])
        return predictions
    
    def prediction_error(self,X,Y):
        predictions = self.predict(X)
        error = np.mean(np.not_equal(predictions,Y))
        return error
        
        