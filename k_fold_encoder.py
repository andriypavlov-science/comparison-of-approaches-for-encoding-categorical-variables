from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        cols=None,
        n_splits=5,
        smooth=20,
        shuffle=True,
        random_state=None
    ):
        self.cols = cols
        self.n_splits = n_splits
        self.smooth = smooth
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        # Строит итоговый mapping по всей обучающей выборке
        """
        Обучение энкодера на всей обучающей выборке.
        Сохраняем статистики для transform().
        """
        # Копируем, чтобы не менять исходный X
        X = X.copy()
        
        # Если cols не заданы, берем все object-колонки
        if self.cols is None:
            self.cols_ = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        else:
            self.cols_ = self.cols

        # Глобальное среднее target
        self.global_mean_ = np.mean(y)
        
        # Здесь будут храниться таблицы соответствий
        self.mapping_ = {}

        # Считаем encoding отдельно для каждого столбца
        for col in self.cols_:
            
            stats = (
                pd.DataFrame({
                    col: X[col],
                    "target": y
                }).groupby(col)["target"].agg(["mean", "count"])
            )
            # Формула сглаживания
            stats["encoding"] = (stats["count"] * stats["mean"] + self.smooth * self.global_mean_ ) / ( stats["count"] + self.smooth )

            #print(stats)
    
            # Сохраняем только готовый mapping
            self.mapping_[col] = stats["encoding"].to_dict()
            # print(self.mapping_[col])
        
        return self
    
    def transform(self, X):
        """
        Кодирование новых данных по статистикам,
        рассчитанным в fit().
        """
        check_is_fitted(
            self,
            ["mapping_", "global_mean_", "cols_"]
        )
        X = X.copy()
        
        for col in self.cols_:
            # Заменяем категории на encoding
            X[col] = X[col].map(self.mapping_[col])
    
            # Неизвестные категории заменяем глобальным средним
            X[col] = X[col].fillna(self.global_mean_)

        return X
    
    def fit_transform(self, X, y):
        """
        K-fold target encoding для обучающей выборки.
        Каждая строка получает encoding,
        рассчитанный без использования самой строки.
        """
        
        X = X.copy()
        
        # Инициализируем колонки
        if self.cols is None:
            self.cols_ = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        else:
            self.cols_ = self.cols

        y = pd.Series(y, index=X.index)

        # Сохраняем глобальное среднее
        self.global_mean_ = y.mean()

        # Итоговый результат
        X_encoded = X.copy()

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        # Обрабатываем каждый столбец отдельно
        for col in self.cols_:
            
            # сюда будут записываться out-of-fold значения
            oof_encoding = pd.Series(
                index=X.index,
                dtype=float
            )

            for train_idx, valid_idx in kf.split(X):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_valid_fold = X.iloc[valid_idx]
        
                # статистики только по train части фолда
                stats = (
                    pd.DataFrame({
                        col: X_train_fold[col],
                        "target": y_train_fold
                    }).groupby(col)["target"].agg(["mean", "count"]))
                
                # сглаживание
                stats["encoding"] = (stats["count"] * stats["mean"] + self.smooth * y_train_fold.mean()) / (stats["count"] + self.smooth)
    
                # кодируем validation часть
                encoded_values = (X_valid_fold[col].map(stats["encoding"]).fillna(y_train_fold.mean()))

                oof_encoding.iloc[valid_idx] = encoded_values

            # заменяем исходный столбец на OOF encoding
            X_encoded[col] = oof_encoding

        # После OOF кодирования сохраняем финальный mapping
        # по всей обучающей выборке для transform()
        self.mapping_ = {}
        
        for col in self.cols_:
            stats = (
            pd.DataFrame({
                col: X[col],
                "target": y
            }).groupby(col)["target"].agg(["mean", "count"]))

            stats["encoding"] = (stats["count"] * stats["mean"] + self.smooth * self.global_mean_) / (stats["count"] + self.smooth)
            self.mapping_[col] = stats["encoding"].to_dict()
        
        return X_encoded

'''
Использование:
 1. Бъем выборку на train и test
 2. Кодируем train:
    X_train_encoded = encoder.fit_transform(X_train, y_train)
 3. Кодируем test часть X ипользуя кодировку на всей train части (без фолдов):
    X_test_encoded = encoder.transform(X_test)

 4. Его можно использовать в Pipeline:

pipeline = Pipeline([
    ("encoder", KFoldTargetEncoder( cols=["city", "color"], n_splits=5, smooth=20 )),
    ("model", RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

 5. Его можно испоьлзовать в cross_validate:

 results = cross_validate(pipeline,X,y,cv=5,scoring="roc_auc")
 Таким образом мы получим вложенные K-fold разбиения
'''