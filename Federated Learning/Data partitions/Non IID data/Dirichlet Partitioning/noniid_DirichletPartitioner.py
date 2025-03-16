from flwr_datasets.partitioner import DirichletPartitioner
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("csv", data_files="Six_Labels_CombinedDataset.csv")

dirichlet_partitioner = DirichletPartitioner(
    num_partitions=4, 
    alpha=0.5, #less alpha means more non-iid nature
    partition_by="activity"
)

dirichlet_partitioner.dataset = dataset['train']

partition0 = dirichlet_partitioner.load_partition(partition_id=0)
partition1 = dirichlet_partitioner.load_partition(partition_id=1)
partition2 = dirichlet_partitioner.load_partition(partition_id=2)
partition3 = dirichlet_partitioner.load_partition(partition_id=3)

print(f"Partition 0 has {len(partition0)} samples")
print(f"Partition 1 has {len(partition1)} samples")
print(f"Partition 2 has {len(partition2)} samples")
print(f"Partition 3 has {len(partition3)} samples")

# Save partitions to CSV files
partition0.to_csv("noniid_Dirichlet_One.csv", index=False)
partition1.to_csv("noniid_Dirichlet_Two.csv", index=False)
partition2.to_csv("noniid_Dirichlet_Three.csv", index=False)
partition3.to_csv("noniid_Dirichlet_Four.csv", index=False)
