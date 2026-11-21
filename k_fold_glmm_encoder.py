from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import KFold
from category_encoders import GLMMEncoder
import pandas as pd
import numpy as np

class KFoldGLMMEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cols=None,
        n_splits=5,
        shuffle=True,
        random_state=None,
        **glmm_params
    ):
        self.cols = cols
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.glmm_params = glmm_params

    def fit(self, X, y):
        """
        Обучение энкодера на всей обучающей выборке.
        """
        
        self.encoder_ = GLMMEncoder(cols=self.cols, **self.glmm_params)
        self.encoder_.fit(X, y)
        
        return self
    
    def transform(self, X):
        """
        Кодирование новых данных по статистикам,
        рассчитанным в fit().
        """
        
        check_is_fitted(self, "encoder_")
        return self.encoder_.transform(X)
    
    def fit_transform(self, X, y):
        X = X.copy()
        y = pd.Series(y, index=X.index)
       
        # Итоговый результат
        X_encoded = X.copy()
       
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        for train_idx, valid_idx in kf.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_valid = X.iloc[valid_idx]

            encoder = GLMMEncoder(cols=self.cols,**self.glmm_params)
            # обучение на train части фолда
            encoder.fit(X_train, y_train)
            
            X_valid_encoded = encoder.transform(X_valid)

            for col in self.cols:
                X_encoded.loc[X_valid.index, col] = X_valid_encoded[col]

        for col in self.cols:
            X_encoded[col] = X_encoded[col].astype(float)
        
        # обучаем финальный encoder на всей выборке
        self.encoder_ = GLMMEncoder(
            cols=self.cols,
            **self.glmm_params
        )
        self.encoder_.fit(X, y)
        
        
        return X_encoded