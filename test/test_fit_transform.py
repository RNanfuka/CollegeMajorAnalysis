import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.fit_transform import fit_transform

# `fit_transform` should return a data frame with same number of rows as the original csv, but
# the correct number of transformed columns, which does not include the drop column, and 
# should be one additional for each unique value for categorical features column.
def test_fit_transform():
    data = pd.read_csv('test/test_df.csv')

    # Separate features and target using the pre-created train/test split
    numeric_features = ['age']
    categorical_features = ['ethnicity']
    ordinal_features = ['height-level']
    binary_features = ['wear-glasses']
    drop_features = ['id']
    target = "pass-class"

    transformed_features = {
        'numeric': numeric_features,
        'categorical': categorical_features,
        'ordinal': ordinal_features,
        'binary': binary_features,
        'drop': drop_features
    }

    # Ordered levels for the education column (lowest to highest)
    ordinal_order = [
        'short',
        'medium',
        'tall'
    ]

    # convert processed data back to DataFrame for better readability
    processed_df = fit_transform(data, transformed_features, ordinal_order, target)
    assert isinstance(processed_df, pd.DataFrame)

    number_unique_features = sum([data[cf].nunique() for cf in categorical_features])
    expected_column_count = (len(numeric_features) + number_unique_features +
                    len(ordinal_features) + len(binary_features) + 1) # 1 for target
    assert processed_df.shape == (data.shape[0], expected_column_count)

# `fit_transform` should throw a ValueError if data frame passed has no data.
def test_fit_transform_empty():
    with pytest.raises(ValueError):
        data = pd.read_csv('test/test_df_empty.csv')
        fit_transform(data, None, None, None)

# `fit_transform` should throw a AttributeError if NoneType is passed for data frame.
def test_fit_transform_error():
    with pytest.raises(AttributeError):
        fit_transform(None, None, None, None)
    

# if __name__ == '__main__':
#     test_fit_transform_empty()