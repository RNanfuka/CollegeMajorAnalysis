import os
import pandas as pd
import click
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.fit_transform import fit_transform

# Main function
@click.command()
@click.option('--input_dir', required=True, help='Path (including filename) to raw data')
@click.option('--out_dir', required=True, help='Path to directory where the preprocessed_adult_train.csv should be saved')
def main(input_dir, out_dir):
    #Input
    adult_df = pd.read_csv(input_dir)

    # Separate features and target using the pre-created train/test split
    numeric_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'native-country']
    ordinal_features = ['education']
    binary_features = ['sex']
    drop_features = ['fnlwgt', 'education-num', 'race']
    target = "income"

    transformed_features = {
        'numeric': numeric_features,
        'categorical': categorical_features,
        'ordinal': ordinal_features,
        'binary': binary_features,
        'drop': drop_features
    }

    # Ordered levels for the education column (lowest to highest)
    education_order = [
        'dropout',
        'HighGrad',
        'CommunityCollege',
        'Bachelors',
        'Masters',
        'Doctorate',
    ]

    # convert processed data back to DataFrame for better readability
    processed_df = fit_transform(adult_df, transformed_features, education_order, target)

    #Output
    if "train" in input_dir:
        processed_df.to_csv(out_dir + "preprocessed_train.csv", index=False)
        click.echo(f"Saved preprocessed_train.csv to file directory {out_dir}")
    elif "test" in  input_dir:
        processed_df.to_csv(out_dir + "preprocessed_test.csv", index=False)
        click.echo(f"Saved preprocessed_test.csv to file directory {out_dir}")
    else:
        processed_df.to_csv(out_dir + "preprocessed_data.csv", index=False)
        click.echo(f"Saved preprocessed_data.csv to file directory {out_dir}")

if __name__ == '__main__':
    main()
