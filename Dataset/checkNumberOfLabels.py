import pandas as pd

partition_path = "./Six_Labels_Datasets/Six_Labels_FourthDataset.csv"

df = pd.read_csv(partition_path)

print("Unique labels in this partition:", df["activity"].unique())  
print("Number of unique labels:", len(df["activity"].unique()))
