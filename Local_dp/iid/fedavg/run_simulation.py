from flwr.simulation import run_simulation
from flwr.server import ServerApp
from flwr.client import ClientApp

from FedAvg_Server import server_fn
from FedAvg_Client1 import client_fn1
from FedAvg_Client2 import client_fn2
from FedAvg_Client3 import client_fn3
from FedAvg_Client4 import client_fn4

from flwr.client.mod import LocalDpMod

local_dp_obj = LocalDpMod(
        clipping_norm=5.0,  
        sensitivity=0.1,
        epsilon=30.0,
        delta=1e-5
)


def combined_client(context):
    node_id=context.node_id
    if node_id%4==0:
        return client_fn1(context)
    elif node_id%4==1:
        return client_fn2(context)
    elif node_id%4==2:
        return client_fn3(context)
    else:
        return client_fn4(context)
    


server_app=ServerApp(server_fn=server_fn)
client_app=ClientApp(client_fn=combined_client,mods=[local_dp_obj])

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=4,
    backend_config={"client_resources":{"num_cpus":1,"num_gpus":0.0}}
)

