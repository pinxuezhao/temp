# coding: utf-8
from device import Devices
import re
import numpy as np
import pickle
    
from autopipe.autopipe.model_profiling import Graph

def get_t5block_idx(name):
    layer_idx = re.findall(r'T5Block\[\d+]', name)
    if len(layer_idx)>0:
        layer_idx = int(layer_idx[0][8:-1])
        return layer_idx
    return None


def get_num_layers(graph):
    num_layers = -1
    for node in graph._nodes:
        name = graph._nodes[node].scope
        layer_idx = get_t5block_idx(name)    
        if layer_idx is not None and layer_idx > num_layers:
            num_layers = layer_idx
    return num_layers


def part_by_t5block(graph, num_partitions):
    num_layers = get_num_layers(graph)+1
    layers_per_partition = int(2*(num_layers)//num_partitions)
    for node in graph._nodes:
        name = graph._nodes[node].scope
        if 'input' in name:
            graph._nodes[node].stage = 0
            continue
        if 'Parameter' in name:
            graph._nodes[node].stage = 0
            continue
        if 'encoder' in name and 'T5Block' not in name:
            graph._nodes[node].stage = 0 
            continue
        if 'encoder' in name and 'T5Block' in name:
            t5block_idx = get_t5block_idx(name)
            graph._nodes[node].stage = t5block_idx//layers_per_partition
            continue
        if 'decoder' in name and 'T5Block' in name:
            t5block_idx = get_t5block_idx(name)
            graph._nodes[node].stage = (num_layers + t5block_idx)//layers_per_partition
            continue
        if 'decoder' in name and 'T5Block' not in name:
            graph._nodes[node].stage = (2*num_layers -1)//layers_per_partition
            continue
        else:
            print(node, name)
#            assert 1==-1
            graph._nodes[node].stage = int((2*num_layers-1)//layers_per_partition)


def get_stage_comp_time(graph, num_partitions):
    forward_comp = [0 for i in range(num_partitions)]
    backward_comp = [0 for i in range(num_partitions)]
    for node in graph._nodes:
        if graph._nodes[node].forward_time:
            forward_comp[graph._nodes[node].stage] += graph._nodes[node].forward_time
        if graph._nodes[node].backward_time:
            backward_comp[graph._nodes[node].stage] += graph._nodes[node].backward_time

    return forward_comp, backward_comp

def get_stage_comm_size(graph, num_partitions):
    pass #now there is no output size infomation in profiled outcome


if __name__ == '__main__':
    num_device = 8
    with open('trace_op', 'rb') as f:
        data = pickle.load(f)
    with open('profile_op', 'rb') as f:
        profile_data  = pickle.load(f)
    with open('profiler_output_op', 'rb') as f:
        output = pickle.load(f)
    test_graph = Graph(None, None, None, None, None)
    test_graph.load_state(profile_data)
    for i in output['forward_times']:
        if len(output['forward_times'][i])>0:
            temp = np.mean(output['forward_times'][i])
            test_graph._nodes[i.id].forward_time = temp
    for i in output['backward_times']:
        if len(output['backward_times'][i])>0:
            temp = np.mean(output['backward_times'][i])
            test_graph._nodes[i.id].backward_time = temp
    for i in output['forward_mem']:
        if len(output['forward_mem'][i])>0:
            temp = np.mean(output['forward_mem'][i])
            test_graph._nodes[i.id].forward_mem = temp
    for i in output['backward_mem']:
        if len(output['backward_mem'][i])>0:
            temp = np.mean(output['backward_mem'][i])
            test_graph._nodes[i.id].backward_mem = temp

    part_by_t5block(test_graph, 8)

    num_micro_batch = int(input("num_micro_batch:"))
    forward_comp, backward_comp = get_stage_comp_time(test_graph, 8)
    forward_comp = [int(i*1000000) for i in forward_comp]
    backward_comp = [int(i*1000000) for i in backward_comp]
    #forward_comm=[3*1000000]*8+[1.5*1000000]*7
    #backward_comm=forward_comm
    forward_comm=[54]*7
    backward_comm=[30]*7
    devices = Devices(num_device, num_micro_batch, forward_comp, backward_comp, forward_comm, backward_comm)
    t = devices.run()
    print("simulated estimated time:"+str(t/1000000))
