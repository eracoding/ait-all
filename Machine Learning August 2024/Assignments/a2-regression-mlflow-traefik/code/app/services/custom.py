from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import mlflow


class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)
            
    def __init__(self, regularization=None, lr=0.001, method='batch', momentum=None, isXavier=False, num_epochs=50, batch_size=32, cv=kfold, l=''):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization

        # For plotting in mlflow
        self.kfold_epoch_mse = []

        # Added xavier initialization flag
        self.isXavier = isXavier

        # Added momentum
        self.momentum = momentum
        self.prev_step = 0

        # For choosing best model upon training
        self.last_r2_score = 0

        self.theta = []

    def mse(self, ytrue, ypred):
        # return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
        return np.mean((ytrue - ypred) ** 2)
    
    def score(self, y_target, yhat):
        ss_res = np.sum((y_target - yhat) ** 2)
        ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
        return 1 - ss_res / ss_tot
    
    def xavier_initialize(self, m):
        # The pseudocode provided solution is below
        # But I dont think it is the correct way to do it
        # Usually it uses:
        # # lower = -np.sqrt(6) / np.sqrt(n_in + n_out)
        # # upper = np.sqrt(6) / np.sqrt(n_in + n_out)

        # But will proceed with the pseudocode
        lower, upper = -1.0 / np.sqrt(m), 1.0 / np.sqrt(m)

        # to get the same results
        np.random.seed(52)

        # numbers = np.random.rand(m)
        numbers = np.random.uniform(lower, upper, m)

        return lower + numbers * (upper - lower)
        # lower = -np.sqrt(6) / np.sqrt(m + 1)  # 1 for output dimension
        # upper = np.sqrt(6) / np.sqrt(m + 1)
        # return np.random.uniform(lower, upper, m)

    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()

        #create a list of kfold scores
        self.kfold_scores = list()
        
        #reset val loss
        self.val_loss_old = np.infty

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            if self.isXavier:
                self.theta = self.xavier_initialize(X_cross_train.shape[1])
            else:
                self.theta = np.zeros(X_cross_train.shape[1])
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__, "xavier_initialization": self.isXavier}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                    # self.learning_rate_decay(epoch)
                   
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]

                    if self.method == 'sgd':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx] 
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)
                    
                    # Appending metrics
                    self.kfold_epoch_mse.append(train_loss)
                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)

                    val_r2_score = self.score(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_r2_score", value=val_r2_score, step=epoch)

                    #early stopping - modified, because it was stopping too early
                    # if np.abs(val_loss_new - self.val_loss_old) < 1e-6:
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break

                    self.val_loss_old = val_loss_new

                self.last_r2_score = val_r2_score
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: {val_loss_new}")
            

    def learning_rate_decay(self, epoch):
        self.lr = self.lr * (0.95 ** (epoch // 10))

    def _train(self, X, y):
        yhat = self.predict(X)
        # print("PREDICTION INSIDE TRAIN: ", yhat.reshape(1, -1))
        m    = X.shape[0]
        grad = (1/m) * X.T @ (yhat - y)
        
        if self.regularization:
            grad += self.regularization.derivation(self.theta)
        
        # Momentum implementation
        if self.momentum and 0 <= self.momentum < 1:
            step = self.lr * grad
            self.theta = self.theta - step + self.momentum * self.prev_step
            self.prev_step = step
        else:
            if self.momentum and self.momentum >= 1:
                print("The value of momentum is more than allowed [0, 1], switching to version without momentum")
            self.theta = self.theta - self.lr * grad
            self.prev_step = 0
        return self.mse(y, yhat)
    
    def predict(self, X, to_transform=False):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    def plot_feature_importance(self, feature_names=None):
        if not hasattr(self, 'theta'):
            raise ValueError("Model coefficients are not available. Fit the model first.")

        # Coefficients
        coefficients = self._coef()
        importance = np.abs(coefficients)

        # Assign default names if feature_names are not provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(1, len(coefficients) + 1)]

        # Sort features by importance
        mask = np.argsort(importance)[::-1]
        sorted_importance = importance[mask]
        sorted_feature_names = np.array(feature_names)[mask]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_feature_names, sorted_importance, color='skyblue')
        plt.xlabel('Coefficient Magnitude (Absolute)')
        plt.title('Feature Importance based on Coefficients')
        plt.gca().invert_yaxis()  # To display the most important feature at the top
        plt.show()
        
        # Test
        # sorted_idx = rf.feature_importances_.argsort()
        # plt.barh(X.columns[sorted_idx], rf.feature_importances_[sorted_idx])
        # plt.xlabel("Random Forest Feature Importance")


class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
        
class Ridge(LinearRegression):
    
    def __init__(self, method, lr, l, momentum, isXavier):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, momentum, isXavier)


class PolynomialRegression(LinearRegression):
    
    def __init__(self, method, lr, l, momentum, isXavier, degree=2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        # Using Ridge as regularization
        self.regularization = RidgePenalty(l)
        
        # Setting the transformer
        self._set_transformer()
        super().__init__(self.regularization, lr, method, momentum, isXavier)
    
    def fit(self, X_train, y_train):
        # Transform the input data to polynomial features
        X_poly = self.poly.fit_transform(X_train)

        # Use the base class's fit method to train the model
        super().fit(X_poly, y_train)

    def _set_transformer(self):
        _test = np.zeros((1, 5))
        self.poly.fit_transform(_test)

    def predict(self, X, to_transform=False):
        X_poly = X
        # Transform the input data to polynomial features before making predictions
        if to_transform:
            X_poly = self.poly.transform(X)
        return super().predict(X_poly)
    
    def save(self, file_path):
        with open(file_path, 'w') as f:
            json.dump({'coefficients': self.theta.tolist()}, f)

    def load(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.theta = np.array(data['coefficients'])

