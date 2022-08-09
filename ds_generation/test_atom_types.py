import pandas as pd
from glob import glob
import sys

parquets_dir = sys.argv[1]

all_parquets = glob(f'{parquets_dir}/*.parquet')


all_df = pd.DataFrame()

for parquet_file in all_parquets:

    all_df = all_df.append(pd.read_parquet(parquet_file))


print(all_df['type'].value_counts())


