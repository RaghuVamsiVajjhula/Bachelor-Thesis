import pandas as pd

df = pd.read_csv('./Original Data/csh104.ann.features.csv')
activities_to_keep = [ 'Cook_Breakfast', 'Bathe', 'Personal_Hygiene', 'Dress', 'Toilet', 'Sleep']
df_filtered = df[df['activity'].isin(activities_to_keep)]
df_filtered.to_csv('Six_Labels_FourthDataset.csv', index=False)
print("Dataset filtered to include only the specified activities.")
