import argparse
import pandas as pd
import math
import os.path as osp

def main():
    parser = argparse.ArgumentParser(
        description="Compute the average of values from multiple CSV files.")
    parser.add_argument('files', nargs='+', help='a list of CSV files')

    args = parser.parse_args()

    # Read all CSV files into a list of DataFrames
    dataframes = [pd.read_csv(file) for file in args.files]

    # Concatenate all DataFrames
    all_data = pd.concat(dataframes)#.T

    base_files = [osp.basename(f) for f in args.files]
    all_data['file'] = base_files
    all_data.set_index('file', inplace=True)
    all_data = all_data.T
    all_data.index.name = 'metric'

    out_filename = 'oquare-agg-metrics.csv'
    all_data.to_csv(out_filename)

    print(f'Read {len(base_files)} files:')
    for f in args.files:
        print(f)
    print(f'Saved aggregated metrics to {out_filename}')

    # all_data = all_data.applymap(lambda x: f"{x:.2f}")
    #
    # max_metric_length = math.ceil(max(map(len, all_data.index.values)) * 1.1)
    #
    # col_identifier = ''.join(['c'] * len(args.files))
    # print("\\begin{tabular}{" + col_identifier + "}")
    # for index, row in all_data.iterrows():
    #     print(f'{index:<{max_metric_length}}', end='')
    #     for value in row.values:
    #         print(f' & {value}', end='')
    #     print(' \\\\')




if __name__ == "__main__":
    main()
