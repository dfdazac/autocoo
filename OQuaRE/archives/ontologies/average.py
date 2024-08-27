import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Compute the average of values from multiple CSV files.")
    parser.add_argument('files', nargs='+', help='a list of CSV files')

    args = parser.parse_args()

    # Read all CSV files into a list of DataFrames
    dataframes = [pd.read_csv(file) for file in args.files]

    # Concatenate all DataFrames
    all_data = pd.concat(dataframes)

    # Compute the average of each column
    average_data = all_data.mean().to_frame().T

    # Transpose the DataFrame
    transposed_data = average_data.T

    # Format the DataFrame with two decimal places
    formatted_data = transposed_data.applymap(lambda x: f"{x:.2f}")

    # Print the resulting DataFrame in LaTeX format
    print("\\begin{tabular}{ll}")
    print("Metric & Average \\\\")
    print("\\hline")
    for index, row in formatted_data.iterrows():
        print(f"{row[0]} \\\\")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
