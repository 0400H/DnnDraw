import dnndraw

def Add_GRU_Cell(dnn, name, node_in, node_h):
    """Add a GRU cell to the graph"""
    graph_name = dnn.add_graph(name, directed=True, subgraph=True)

    # Gates
    reset_gate = f"{name}_Rt"
    update_gate = f"{name}_Zt"
    candidate_state = f"{name}_Nt"
    hidden_state = f"{name}_Ht"

    sub_graph_name = dnn.add_graph(graph_name+"_0", directed=True, subgraph=True, graph_attr={'style':'invis'})
    # Add reset_gate
    dnn.add_node(
        in_nodes=[node_in, node_h],
        node_info={
            'name': reset_gate,
            'type': 'reset_gate Rt',
            'formula': 'Matmul(W_ir,It)+Matmul(W_hr,H[t-1])+Br',
            'Shape W_ir': '[hidden_size, input_size]',
            'Shape W_hr': '[hidden_size, hidden_size]',
            'Shape out': ['batch', 'hidden_size']
        },
        graph_name=sub_graph_name,
    )

    # Add update_gate
    dnn.add_node(
        in_nodes=[node_in, node_h],
        node_info={
            'name': update_gate,
            'type': 'update_gate Zt',
            'formula': 'Matmul(W_iz,It)+Matmul(W_hz,H[t-1])+Bz',
            'Shape W_iz': '[hidden_size, input_size]',
            'Shape W_hz': '[hidden_size, hidden_size]',
            'Shape out': ['batch', 'hidden_size']
        },
        graph_name=sub_graph_name,
    )

    # Add candidate state
    dnn.add_node(
        in_nodes=[node_in, node_h, reset_gate],
        node_info={
            'name': candidate_state,
            'type': 'candidate_state Nt',
            'formula': 'tanh(Matmul(W_in,It)+B_in+EleWiseMul(Rt,Matmul(W_hn,H[t-1])+B_hn))',
            'Shape W_in': '[hidden_size, input_size]',
            'Shape W_hn': '[hidden_size, hidden_size]',
            'Shape out': ['batch', 'hidden_size']
        },
        graph_name=graph_name,
    )

    # Add hidden state
    dnn.add_node(
        in_nodes=[node_h, update_gate, candidate_state],
        node_info={
            'name': hidden_state,
            'type': 'hidden_state Ht',
            'formula': 'EleWiseMul((1-Zt),Nt)+EleWiseMul(Zt,H[t-1])',
            'Shape out': ['batch', 'hidden_size']
        },
        graph_name=graph_name,
    )

    dnn.merge_subgraph(graph_name, sub_graph_name)
    dnn.merge_subgraph(dnn.name, graph_name)
    return hidden_state

# https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
# https://d2l.ai/chapter_recurrent-modern/gru.html
def GRU(name, seq_len=1, num_layers=1):
    # Create main graph
    dnn = dnndraw.graph(name=name, layout="BT")

    it_cache = [None for _ in range(seq_len)]
    ht_cache = [None for _ in range(num_layers)]

    # initial input data
    for i in range(seq_len):
        it = dnn.add_node(
            in_nodes=[],
            node_info={
                'name': f'in_t{i}',
                'type': 'Input_Data',
                'shape': ['batch', 'input_size']
            }
        )
        it_cache[i] = it
    # Add initial states
    for i in range(num_layers):
        ht = dnn.add_node(
            in_nodes=[],
            node_info={
                'name': f'h_l{i}',
                'type': 'Initial_Hidden',
                'shape': ['batch', 'proj_size']
            }
        )
        ht_cache[i] = ht
    # Add GRU cell
    for i in range(num_layers):
        for j in range(seq_len):
            ht = Add_GRU_Cell(
                dnn,
                name=f'Cell_l{i}_t{j}',
                node_in=it_cache[j],
                node_h=ht_cache[i],
            )
            it_cache[j] = ht
            ht_cache[i] = ht
    return dnn

if __name__ == '__main__':
    dnn = GRU("GRU Cell", seq_len=1, num_layers=1)
    print(dnn.source())
    dnn.export(format='svg')
