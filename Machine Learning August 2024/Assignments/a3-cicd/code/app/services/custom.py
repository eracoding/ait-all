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


class RidgePenalty:

    def __init__(self, l):
        self.l = l

    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta


class LogisticRegression:

    def __init__(self, k, n, method='batch', batch_size=512, alpha=0.001, max_iter=5000, regularization=True, l=0.1):
        self.k = k
        self.n = n
        self.method = method
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size

        self.l = l

        self.reg_flag = regularization
        self.regularization = RidgePenalty(self.l) if self.reg_flag else None

        self.W = None


    def fit(self, X, Y):
        if self.method not in ["batch", "minibatch", "sto"]:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')

        # Creating new set of weights each time fit function is called is bad idea
        # probably we do not want to do it that way -> initialization during init? - for kfold, introduce flag?
        np.random.seed(42)
        self.W = np.random.rand(self.n, self.k)

        self.losses = []

        Y = np.eye(self.k)[Y]
        # with mlflow.start_run(run_name=f"{type(self).__name__}", nested=True):
        #     params = {"method": self.method, "lr": self.alpha, "reg": type(self).__name__, "regularization": self.reg_flag}
        #     mlflow.log_params(params=params)
        self.val_loss_old = np.infty

        if self.method == "batch":
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.losses.append(loss)

                self.W = self.W - self.alpha * grad

                # if i % 10 == 0:
                    # print(f"Loss at iteration {i}", loss)

                    # if i == 30:
                    #    self.alpha = 0.0009025
                if i % 125 == 0:
                    self.learning_rate_decay(i)

                    # print(f'Learning rate: ', self.alpha)

                mlflow.log_metric(key="train_loss", value=loss, step=i)

        elif self.method == "minibatch":
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0]) #<----with replacement
                batch_X = X[ix:ix+self.batch_size]
                batch_Y = Y[ix:ix+self.batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 10 == 0:
                    # print(f"Loss at iteration {i}", loss)

                    if i % 90 == 0:
                        self.learning_rate_decay(i)
                    # print(f'Learning rate: ', self.alpha)
                mlflow.log_metric(key="train_loss", value=loss, step=i)

        elif self.method == "sto":
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]

                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)

                self.W = self.W - self.alpha * grad

                # if i % 500 == 0:
                #     print(f"Loss at iteration {i}", loss)

                mlflow.log_metric(key="train_loss", value=loss, step=i)

        return self


        # if np.allclose(loss, self.val_loss_old):
        #     print(f"Break - Loss at iteration {i}", loss)

        # self.val_loss_old = loss

        # print(f"Time taken: {time.time() - start_time}")

    def learning_rate_decay(self, epoch, min_lr=1e-8):
        new_alpha = self.alpha * (0.95 ** (epoch // 10))
        # new_alpha = self.alpha * 0.1
        self.alpha = max(new_alpha, min_lr)

    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)

    def score(self, X, Y):
        """Score method required by GridSearchCV."""
        Y_pred = self.predict(X)
        accuracy = np.mean(Y_pred == Y)
        return accuracy

    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)

        loss = - np.sum(Y * np.log(h)) / m

        if self.reg_flag:
            loss += self.regularization(self.W) / (m) # Divide by 2*m to normalize

        error = h - Y

        grad = self.softmax_grad(X, error)

        if self.reg_flag:
            grad += (self.regularization.derivation(self.W) / m)

        return loss, grad

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return X.T @ error

    def h_theta(self, X, W):
        return self.softmax(X @ W)

    # For performing GridSearch
    def get_params(self, deep=True):
        """Get the parameters for GridSearchCV."""
        return {
            'k': self.k,
            'n': self.n,
            'method': self.method,
            'batch_size': self.batch_size,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'regularization': self.reg_flag,
            'l': self.l
        }
    # For performing GridSearch
    def set_params(self, **params):
        """Set the parameters for GridSearchCV."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def plot(self):
        plt.plot(np.arange(len(self.losses)), self.losses, label = "Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()

    # Task 1
    @staticmethod
    def accuracy(y_true, y_pred):
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)

        return correct_predictions / total_predictions

    @staticmethod
    def precision(y_true, y_pred, class_label):
        TP = np.sum((y_true == class_label) & (y_pred == class_label))
        FP = np.sum((y_true != class_label) & (y_pred == class_label))

        return TP / (TP + FP) if TP + FP > 0 else 0.0

    @staticmethod
    def recall(y_true, y_pred, class_label):
        TP = np.sum((y_true == class_label) & (y_pred == class_label))
        FN = np.sum((y_true == class_label) & (y_pred != class_label))

        return TP / (TP + FN) if TP + FN > 0 else 0.0

    @staticmethod
    def f1_score(y_true, y_pred, class_label):
        P = LogisticRegression.precision(y_true, y_pred, class_label)
        R = LogisticRegression.recall(y_true, y_pred, class_label)

        return 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    @staticmethod
    def macro_precision(y_true, y_pred):
        classes = np.unique(y_true)
        precisions = [LogisticRegression.precision(y_true, y_pred, class_label) for class_label in classes]

        return np.mean(precisions)

    @staticmethod
    def macro_recall(y_true, y_pred):
        classes = np.unique(y_true)
        recalls = [LogisticRegression.recall(y_true, y_pred, class_label) for class_label in classes]

        return np.mean(recalls)

    @staticmethod
    def macro_f1(y_true, y_pred):
        classes = np.unique(y_true)
        f1_scores = [LogisticRegression.f1_score(y_true, y_pred, class_label) for class_label in classes]

        return np.mean(f1_scores)

    @staticmethod
    def weighted_precision(y_true, y_pred):
        classes = np.unique(y_true)
        total_samples = len(y_true)

        weights = [(np.sum(y_true == class_label) / total_samples) for class_label in classes]
        precisions = [LogisticRegression.precision(y_true, y_pred, class_label) for class_label in classes]

        return np.sum([precision * w_coef for precision, w_coef in zip(precisions, weights)])

    @staticmethod
    def weighted_recall(y_true, y_pred):
        classes = np.unique(y_true)
        total_samples = len(y_true)

        weights = [(np.sum(y_true == class_label) / total_samples) for class_label in classes]
        recalls = [LogisticRegression.recall(y_true, y_pred, class_label) for class_label in classes]

        return np.sum([recall * w_coef for recall, w_coef in zip(recalls, weights)])

    @staticmethod
    def weighted_f1(y_true, y_pred):
        classes = np.unique(y_true)
        total_samples = len(y_true)

        weights = [(np.sum(y_true == class_label) / total_samples) for class_label in classes]
        f1_scores = [LogisticRegression.f1_score(y_true, y_pred, class_label) for class_label in classes]

        return np.sum([f1_score * w_coef for f1_score, w_coef in zip(f1_scores, weights)])

    @staticmethod
    def classification_report(y_true, y_pred):
        classes = np.unique(y_true)
        report = []
        header = f"{'Class':<16}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<10}"
        report.append(header)
        report.append("=" * len(header))

        for class_label in classes:
            precision = LogisticRegression.precision(y_true, y_pred, class_label)
            recall = LogisticRegression.recall(y_true, y_pred, class_label)
            f1 = LogisticRegression.f1_score(y_true, y_pred, class_label)
            support = np.sum(y_true == class_label)

            report.append(
                f"{class_label:<16}{precision:<12.4f}{recall:<12.4f}{f1:<12.4f}{support:<10}"
            )

        report.append("=" * len(header))
        macro_precision = LogisticRegression.macro_precision(y_true, y_pred)
        macro_recall = LogisticRegression.macro_recall(y_true, y_pred)
        macro_f1 = LogisticRegression.macro_f1(y_true, y_pred)

        weighted_precision = LogisticRegression.weighted_precision(y_true, y_pred)
        weighted_recall = LogisticRegression.weighted_recall(y_true, y_pred)
        weighted_f1 = LogisticRegression.weighted_f1(y_true, y_pred)

        accuracy = LogisticRegression.accuracy(y_true, y_pred)
        total_support = len(y_true)

        report.append(
            f"{'Accuracy':<16}{'':<12}{'':<12}{accuracy:<12.4f}{total_support:<10}"
        )

        report.append(
            f"{'Macro Avg':<16}{macro_precision:<12.4f}{macro_recall:<12.4f}{macro_f1:<12.4f}{total_support:<10}"
        )
        report.append(
            f"{'Weighted Avg':<16}{weighted_precision:<12.4f}{weighted_recall:<12.4f}{weighted_f1:<12.4f}{total_support:<10}"
        )

        return "\n".join(report)

    # For understanding which feature is important
    def plot_feature_importance(self, feature_names=None):
        if not hasattr(self, 'W'):
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

        sorted_feature_names = sorted_feature_names.tolist() if isinstance(sorted_feature_names, np.ndarray) else sorted_feature_names
        sorted_importance = sorted_importance.tolist() if isinstance(sorted_importance, np.ndarray) else sorted_importance

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_feature_names, sorted_importance, color='skyblue')
        plt.xlabel('Coefficient Magnitude (Absolute)')
        plt.title('Feature Importance based on Coefficients')
        plt.gca().invert_yaxis()  # To display the most important feature at the top
        plt.show()

    def _coef(self):
        return self.W


    def save(self, file_path):
        with open(file_path, 'w') as f:
            json.dump({'coefficients': self.W.tolist()}, f)

    def load(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.W = np.array(data['coefficients'])
