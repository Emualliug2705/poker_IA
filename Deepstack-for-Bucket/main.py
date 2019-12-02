from Generate_data import *
from Cluster_data import *
from Cluster_result import *

# Generate data
for s in ['river', 'turn', 'flop']:
    data = Gen_data(street=s, file_path=settings.file_path)
    data.generate_data(heuristic=settings.dict_heuristic)
    cluster_data_main(street=s, cluster_k=settings.dict_clusters[s])

# Generate results
for s in ['flop', 'turn', 'river']:
    cluster_result = CLUSTER_RESULT(street=s, file_path='data/', turn_sample_count=20,river_sample_count=10, opponent_sample_count=20,
                                    comb_flag=True, normalize_flag=True)
    cluster_result.computer_cluster_result()
