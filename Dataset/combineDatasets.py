import pandas as pd
import os

folder_path="./Six_Labels_Datasets"

csv_files=[file for  file in os.listdir(folder_path) if file.endswith(".csv")]

dataframes=[]

for file in csv_files:
    file_path=os.path.join(folder_path,file)
    df=pd.read_csv(file_path)
    dataframes.append(df)


combined_df=pd.concat(dataframes, ignore_index=True)


combined_df.drop_duplicates(inplace=True)

output_path="Six_Labels_CombinedDataset.csv"
combined_df.to_csv(output_path,index=False)

print(f"Combined dataset saved as {output_path}")

