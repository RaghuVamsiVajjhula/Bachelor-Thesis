import flwr as fl
from flwr.server.strategy import Krum
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import os

accuracy_history = []

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    total_examples = sum([num_examples for num_examples, _ in metrics])
    weighted_metrics = {}

    for metric in metrics[0][1].keys():
        weighted_sum = sum([m[metric] * num_examples for num_examples, m in metrics])
        weighted_metrics[metric] = weighted_sum / total_examples

    return weighted_metrics

def plot_server_accuracy(accuracy_history, num_rounds,save_path="server_plots/accuracy_point.txt"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_rounds + 1), accuracy_history, marker='o', color='blue', linewidth=2)
    plt.title('Server Model Accuracy Over Rounds', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Round', fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('server_plots', exist_ok=True)
    plt.savefig('server_plots/server_accuracy.png')
    plt.show()

    with open(save_path, "w") as f:
        for acc in accuracy_history:
            f.write(f"{acc}\n")

class AggregateKrumStrategy(Krum):
    def __init__(self, num_malicious_clients: int, *args, **kwargs):
        self.num_rounds = kwargs.pop('num_rounds', 101)
        super().__init__(num_malicious_clients=num_malicious_clients, *args, **kwargs)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        if not results:
            return None, {}

        aggregated_metrics = weighted_average(
            [(r.num_examples, r.metrics) for _, r in results]
        )
        
        if "accuracy" in aggregated_metrics:
            accuracy_history.append(aggregated_metrics["accuracy"])
            
        print(f"Round {server_round} aggregated metrics:")  
        for metric, value in aggregated_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        if server_round == self.num_rounds:
            print("\nFederated Learning completed. Generating accuracy graph...")
            # plot_server_accuracy(accuracy_history, len(accuracy_history))
            filename="server_plots/accuracy_point.txt"
            plot_server_accuracy(accuracy_history,len(accuracy_history),save_path=filename)


        return aggregated_metrics.get("loss", None), aggregated_metrics

num_rounds = 101



# fl.server.start_server(
#     server_address="0.0.0.0:8080",
#     config=fl.server.ServerConfig(num_rounds=num_rounds),
#     strategy=strategy
# )

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, start_server

def server_fn(context:Context):
    strategy = AggregateKrumStrategy(
    num_malicious_clients=1,  
    num_clients_to_keep=3,    
    fraction_fit=0.5,
    fraction_evaluate=0.5,
    min_fit_clients=4,
    min_evaluate_clients=4,
    min_available_clients=4,
    num_rounds=num_rounds
    )

    config=fl.server.ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(config=config,strategy=strategy)

if __name__=="__main__":
    app=ServerApp(server_fn=server_fn)
    
