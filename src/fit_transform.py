import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def get_feature_names(preprocessor):
    feature_names = []

    # Ordinal features
    ordinal_features = preprocessor.transformers_[0][2]
    feature_names.extend(ordinal_features)

    # Binary features
    binary_features = preprocessor.transformers_[1][2]
    binary_encoder = preprocessor.transformers_[1][1].named_steps['encoder']
    binary_feature_names = binary_encoder.get_feature_names_out(binary_features)
    feature_names.extend(binary_feature_names)

    # Categorical features
    categorical_features = preprocessor.transformers_[2][2]
    categorical_encoder = preprocessor.transformers_[2][1].named_steps['encoder']
    categorical_feature_names = categorical_encoder.get_feature_names_out(categorical_features)
    feature_names.extend(categorical_feature_names)

    # Numeric features
    numeric_features = preprocessor.transformers_[3][2]
    feature_names.extend(numeric_features)

    return feature_names

def fit_transform(df, transformed_features, ordinal_order, target):
    """
    Transform the given DataFrame (df) based on given transformation required for each column
    and return the processed DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        The data to be transformed.
    transformed_features : dict
        a dictionary that contains the following keys: 'numeric', 'categorical', 'ordinal', 'binary',
        and 'drop'. Each maps to a list of column names to be transformed in that fashion.
    ordinal_order : list
        a list that contains the unique value of the `transformed_features['ordinal']` column in
        the order to be transformed by OrdinalEncoder.
    target: str
        the column name in df to be used as target.

    Returns:
    -------
    pandas.DataFrame
        the preprocessed DataFrame

    Examples:
    --------
    >>> data = pd.read_csv('test/test_df.csv')
    >>> numeric_features = ['age']
    >>> categorical_features = ['ethnicity']
    >>> ordinal_features = ['height-level']
    >>> binary_features = ['wear-glasses']
    >>> drop_features = ['id']
    >>> target = "pass-class"
    >>> transformed_features = {
        'numeric': numeric_features,
        'categorical': categorical_features,
        'ordinal': ordinal_features,
        'binary': binary_features,
        'drop': drop_features
    }

    >>> ordinal_order = [
        'short',
        'medium',
        'tall'
    ]
    >>> processed_df = fit_transform(data, transformed_features, ordinal_order, target)
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot be transformed.")

    numeric_features = transformed_features['numeric']
    categorical_features = transformed_features['categorical']
    ordinal_features = transformed_features['ordinal']
    binary_features = transformed_features['binary']
    drop_features = transformed_features['drop']

    # Separate features and target using the pre-created train/test split
    X = df.drop(columns = drop_features + [target])
    y_train = df[target]

    # Column Preprocessing
    # Pipelines for different data types
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ordinal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[ordinal_order], handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    binary_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', ordinal_pipeline, ordinal_features),
            ('binary', binary_pipeline, binary_features),
            ('categorical', categorical_pipeline, categorical_features),
            ('numeric', numeric_pipeline, numeric_features),
        ]
    )

    # generate preprocessed training data just to check how it looks
    X_processed = preprocessor.fit_transform(X)

    # get the feature names after preprocessing
    feature_names = get_feature_names(preprocessor)

    # convert processed data back to DataFrame for better readability
    processed_df = pd.DataFrame(X_processed, columns=feature_names)
    processed_df[target] = y_train.reset_index(drop=True)
    return processed_df