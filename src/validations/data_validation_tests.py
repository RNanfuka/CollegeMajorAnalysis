import argparse
import pandas as pd
from check_data_file_format import *
from check_correlations import validate_correlation_schema

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset correlations with pandera.")
    parser.add_argument(
        "--data",
        default="data/adult.csv",
        help="Path to CSV file to validate (default: data/adult.csv)",
    )
    file_path = parser.parse_args().data
    test_check_data_file_format(file_path)
    df = pd.read_csv(file_path)
    validate_correlation_schema(df)

if __name__ == "__main__":
    main()