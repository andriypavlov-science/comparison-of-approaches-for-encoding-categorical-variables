import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin
from k_fold_encoder import KFoldTargetEncoder
from k_fold_glmm_encoder import KFoldGLMMEncoder
from skrub import SimilarityEncoder, MinHashEncoder
from sklearn.compose import ColumnTransformer

class MetaEncoder(BaseEstimator, TransformerMixin):
    """
    Мета-кодировщик, объединяющий category_encoders и skrub (MinHash, Similarity).
    """

    def __init__(self, encode_columns_as=None, custom_mapping=None):
        self.encode_columns_as = encode_columns_as
        self.custom_mapping = custom_mapping

    def _get_default_mapping(self, cols):
        """
        Возвращает словарь-маппинг, где каждое значение — это callable (лямбда/функция),
        возвращающая готовый экземпляр трансформера для заданных cols.
        """
        return {
            # category_encoders
            "OneHot": lambda: ce.OneHotEncoder(cols=cols),
            "Hash": lambda: ce.HashingEncoder(cols=cols),
            "Binary": lambda: ce.BinaryEncoder(cols=cols),
            "Ordinal": lambda: ce.OrdinalEncoder(cols=cols),
            "RankHot": lambda: ce.RankHotEncoder(cols=cols),
            # skrub через ColumnTransformer
            "Similarity": lambda: ColumnTransformer(
                [("cat", SimilarityEncoder(), cols)],
                remainder="drop"  # Оставляем drop, так как passthrough-колонки мы обработаем отдельно
            ),
            "MinHash": lambda: ColumnTransformer(
                [
                    (f"minhash_{col}", MinHashEncoder(), [col])
                    for col in cols
                ],
                remainder="drop"
            ),
            'Leave-one-out': lambda: ce.LeaveOneOutEncoder(cols=cols),
            'K-fold': lambda: KFoldTargetEncoder(cols=cols),
            "CatBoost": lambda: ce.CatBoostEncoder(cols=cols),
            "CV-GLMM": lambda: KFoldGLMMEncoder(cols=cols),
            "WOE": lambda: ce.WOEEncoder(cols=cols),
        }

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = X.columns.tolist()
        self.encoders_ = {}

        if not self.encode_columns_as:
            self.other_cols_ = self.feature_names_in_
            return self

        encoded_cols = []

        for enc_name, cols in self.encode_columns_as.items():
            if not cols:
                continue

            # Получаем стандартный маппинг функций-фабрик
            mapping = self._get_default_mapping(cols)

            if self.custom_mapping and enc_name in self.custom_mapping:
                # Если передан кастомный конструктор
                encoder_instance = self.custom_mapping[enc_name](cols)
            elif enc_name in mapping:
                # Создаем экземпляр по имени
                encoder_instance = mapping[enc_name]()
            else:
                raise ValueError(f"Неизвестный кодировщик '{enc_name}'")

            # Обучаем трансформер
            encoder_instance.fit(X[cols], y)

            self.encoders_[enc_name] = {
                "instance": encoder_instance,
                "cols": list(cols)
            }
            encoded_cols.extend(cols)

        # Выделяем необработанные столбцы
        self.other_cols_ = [col for col in self.feature_names_in_ if col not in encoded_cols]

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        transformed_parts = []

        for enc_name, item in self.encoders_.items():
            enc_instance = item["instance"]
            cols = item["cols"]

            # Применяем трансформацию
            part = enc_instance.transform(X[cols])

            # ColumnTransformer и skrub могут возвращать numpy.ndarray или sparse matrix
            if hasattr(part, "toarray"):  # Разреживание (sparse)
                part = part.toarray()

            if not isinstance(part, pd.DataFrame):
                # Если вернулся numpy array, пытаемся извлечь имена колонок
                if hasattr(enc_instance, "get_feature_names_out"):
                    col_names = enc_instance.get_feature_names_out()
                else:
                    col_names = [f"col_{i}" for i in range(part.shape[1])]

                part = pd.DataFrame(part, index=X.index, columns=col_names)

            # Добавляем префикс имени кодировщика
            part = part.add_prefix(f"{enc_name}_")
            transformed_parts.append(part)

        # Добавляем необработанные столбцы
        if self.other_cols_:
            transformed_parts.append(X[self.other_cols_])

        if transformed_parts:
            return pd.concat(transformed_parts, axis=1)
        return X.copy()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

#
# df = pd.DataFrame({
#     "category": ["A", "B", "A", "C"],
#     "gender": ["M", "F", "F", "M"],
#     "status": ["low", "medium", "high", "low"],
#     "age": [25, 30, 45, 35]
# })
#
# encode_columns_as = {
#     "OneHot": ["category"],
#     "Similarity": ["gender"],
#     "MinHash": ["status"]
# }
#
# meta_enc = MetaEncoder(encode_columns_as=encode_columns_as)
# df_encoded = meta_enc.fit_transform(df)
#
# print(df_encoded.columns.tolist())