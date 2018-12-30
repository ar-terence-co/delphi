# delphi

Netowrk Alpha, Beta, Gamma -> Failed

# Network Delta [in:64 res:64,128,256,512 fc:1024]

# code: network_delta10
# 10 Epochs, no additional (OVERFITTED. Try reducing epochs)
# training
# {
# 'accuracy_t0.3': 0.78869355,
# 'accuracy_t0.5': 0.833752,
# 'accuracy_t0.7': 0.85663784,
# 'global_step': 7055,
# 'loss': 0.4029069,
# 'precision_t0.3': 0.65378237,
# 'precision_t0.5': 0.7350073,
# 'precision_t0.7': 0.8107618,
# 'recall_t0.3': 0.90016073,
# 'recall_t0.5': 0.8544716,
# 'recall_t0.7': 0.7942296
# }
# testing
# {
# 'accuracy_t0.3': 0.41414228,
# 'accuracy_t0.5': 0.46274576,
# 'accuracy_t0.7': 0.5174247,
# 'global_step': 7055,
# 'loss': 1.6287609,
# 'precision_t0.3': 0.2283831,
# 'precision_t0.5': 0.22642933,
# 'precision_t0.7': 0.22247066,
# 'recall_t0.3': 0.6448905,
# 'recall_t0.5': 0.5478102,
# 'recall_t0.7': 0.43576643
# }

# code: network_delta50
# 50 Epochs, no additional (OVERFITTED. Try reducing epochs)
# training
# {
# 'accuracy_t0.3': 0.93314105,
# 'accuracy_t0.5': 0.92151207,
# 'accuracy_t0.7': 0.9042081,
# 'global_step': 35271,
# 'loss': 0.23196791,
# 'precision_t0.3': 0.94550484,
# 'precision_t0.5': 0.96273416,
# 'precision_t0.7': 0.9740443,
# 'recall_t0.3': 0.8675861,
# 'recall_t0.5': 0.81749725,
# 'recall_t0.7': 0.75886285
# }
# testing
# {
# 'accuracy_t0.3': 0.6677074,
# 'accuracy_t0.5': 0.69319046,
# 'accuracy_t0.7': 0.7152139,
# 'global_step': 35271,
# 'loss': 1.8804364,
# 'precision_t0.3': 0.12468672,
# 'precision_t0.5': 0.13278688,
# 'precision_t0.7': 0.1574973,
# 'recall_t0.3': 0.07262774,
# 'recall_t0.5': 0.059124086,
# 'recall_t0.7': 0.05328467
# }

# Network Delta [in:32 res:32,64,128,256 fc:1024] Data 2006-2016

# code: network_delta10_e5_f0.5 (NOT FULLY TRAINED)
# 5 Epochs, All res filters are reduced by half , no additional
# training
# {
# 'accuracy_t0.3': 0.57961124,
# 'accuracy_t0.5': 0.6922053,
# 'accuracy_t0.7': 0.7365706,
# 'global_step': 6573,
# 'loss': 0.6350544,
# 'precision_t0.3': 0.45419472,
# 'precision_t0.5': 0.5553773,
# 'precision_t0.7': 0.66851807,
# 'recall_t0.3': 0.8583171,
# 'recall_t0.5': 0.7074394,
# 'recall_t0.7': 0.5254017
# }
# testing
# {
# 'accuracy_t0.3': 0.34022444,
# 'accuracy_t0.5': 0.4675555,
# 'accuracy_t0.7': 0.59471774,
# 'global_step': 6573,
# 'loss': 1.0061787,
# 'precision_t0.3': 0.2001417,
# 'precision_t0.5': 0.17953321,
# 'precision_t0.7': 0.17532264,
# 'recall_t0.3': 0.6186131,
# 'recall_t0.5': 0.3649635,
# 'recall_t0.7': 0.20328467
# }

# code: network_delta10_e10_f0.5 (SWEET SPOT BUT NO THERE YET)
# 10 Epochs, All res filters are reduced by half , no additional
# training
# {
# 'accuracy_t0.3': 0.7317613,
# 'accuracy_t0.5': 0.727834,
# 'accuracy_t0.7': 0.71498704,
# 'global_step': 13146,
# 'loss': 0.7832063,
# 'precision_t0.3': 0.6817968,
# 'precision_t0.5': 0.74339694,
# 'precision_t0.7': 0.80745083,
# 'recall_t0.3': 0.4715334,
# 'recall_t0.5': 0.36727965,
# 'recall_t0.7': 0.26873782
# }
# testing
# {
# 'accuracy_t0.3': 0.7433128,
# 'accuracy_t0.5': 0.7621298,
# 'accuracy_t0.7': 0.76753014,
# 'global_step': 13146,
# 'loss': 1.0327832,
# 'precision_t0.3': 0.2625786,
# 'precision_t0.5': 0.3753943,
# 'precision_t0.7': 0.46411484,
# 'recall_t0.3': 0.060948905,
# 'recall_t0.5': 0.043430656,
# 'recall_t0.7': 0.03540146
# }

# code: network_delta10_e20_f0.5 (ABOUT TO OVERFIT)
# 20 Epochs, All res filters are reduced by half , no additional
# training
# {
# 'accuracy_t0.3': 0.8426579,
# 'accuracy_t0.5': 0.8165313,
# 'accuracy_t0.7': 0.7851295,
# 'global_step': 26291,
# 'loss': 0.58255965,
# 'precision_t0.3': 0.9260069,
# 'precision_t0.5': 0.952701,
# 'precision_t0.7': 0.97042775,
# 'recall_t0.3': 0.60964054,
# 'recall_t0.5': 0.5135135,
# 'recall_t0.7': 0.41297483
# }
# testing
# {
# 'accuracy_t0.3': 0.7205299,
# 'accuracy_t0.5': 0.7448317,
# 'accuracy_t0.7': 0.7597671,
# 'global_step': 26291,
# 'loss': 1.4893969,
# 'precision_t0.3': 0.21626984,
# 'precision_t0.5': 0.25174826,
# 'precision_t0.7': 0.3257329,
# 'recall_t0.3': 0.079562046,
# 'recall_t0.5': 0.052554745,
# 'recall_t0.7': 0.03649635
# }


