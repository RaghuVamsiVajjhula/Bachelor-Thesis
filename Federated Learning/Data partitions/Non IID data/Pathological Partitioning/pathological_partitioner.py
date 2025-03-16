from flwr_datasets.partitioner import PathologicalPartitioner
import pandas as pd
from datasets import load_dataset

dataset=load_dataset("csv",data_files="Six_Labels_CombinedDataset.csv")

partitioner=PathologicalPartitioner(num_classes_per_partition=6,partition_by="activity",num_partitions=4)


partitioner.dataset=dataset['train']


partition=partitioner.load_partition(partition_id=0)
partition1=partitioner.load_partition(partition_id=1)
partition2=partitioner.load_partition(partition_id=2)
partition3=partitioner.load_partition(partition_id=3)

print(f"Partition 0 has {len(partition)} samples")
print(f"Partition 1 has {len(partition1)} samples")
print(f"Partition 2 has {len(partition2)} samples")
print(f"Partition 3 has {len(partition3)} samples")

partition.to_csv("noniid_Pathological_One.csv",index=False)
partition1.to_csv("noniid_Pathological_Two.csv",index=False)
partition2.to_csv("noniid_Pathological_Three.csv",index=False)
partition3.to_csv("noniid_Pathological_Four.csv",index=False)

