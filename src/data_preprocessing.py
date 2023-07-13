import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

numeric_features = ['num_fotos', 'ano_de_fabricacao', 'ano_modelo', 'hodometro']


categorical_features = ['marca', 'modelo', 'versao', 'cambio', 'num_portas',
                        'tipo', 'blindado', 'cor', 'tipo_vendedor', 'cidade_vendedor',
                        'estado_vendedor', 'anunciante', 'entrega_delivery', 'troca',
                        'elegivel_revisao', 'dono_aceita_troca', 'veiculo_único_dono',
                        'revisoes_concessionaria', 'ipva_pago', 'veiculo_licenciado',
                        'garantia_de_fábrica', 'revisoes_dentro_agenda', 'veiculo_alienado']

def preprocess_data(data):
    """
    Preprocesses the input data.

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    logging.info("Preprocessing data...")

    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])

    preprocessed_data = preprocessor.fit_transform(data)
    logging.info("Preprocessing complete.")
    return pd.DataFrame(preprocessed_data, columns=preprocessor.get_feature_names_out())

