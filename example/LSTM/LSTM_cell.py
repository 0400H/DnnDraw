import dnndraw

def Add_LSTM_Cell(dnn, name, node_in, node_h, node_c):
    """Add a LSTM cell to the graph"""
    graph_name = dnn.add_graph(name, directed=True, subgraph=True)

    # Gates
    forget_gate = f"{name}_Ft"
    input_gate = f"{name}_It"
    output_gate = f"{name}_Ot"
    cell_update = f"{name}_ct"
    cell_state = f"{name}_Ct"
    hidden_state = f"{name}_Ht"

    sub_graph_name0 = dnn.add_graph(graph_name+'_0', directed=True, subgraph=True, graph_attr={'style':'invis'})
    # Add forget gate
    dnn.add_node(
        in_nodes=[node_in, node_h],
        node_info={
            'name': forget_gate,
            'type': 'forget_gate Ft',
            'formula': 'Matmul(W_if,It)+Matmul(W_hf,H[t-1])+Bf',
            'Shape W_if': '[hidden_size, input_size]',
            'Shape W_hf': '[hidden_size, Proj_size]',
            'Shape out': ['batch', 'hidden_size']
        },
        graph_name=sub_graph_name0,
    )

    # Add input gate
    dnn.add_node(
        in_nodes=[node_in, node_h],
        node_info={
            'name': input_gate,
            'type': 'input_gate It',
            'formula': 'Matmul(W_ii,It)+Matmul(W_hi,H[t-1])+Bi',
            'Shape W_ii': '[hidden_size, input_size]',
            'Shape W_hi': '[hidden_size, Proj_size]',
            'Shape out': ['batch', 'hidden_size']
        },
        graph_name=sub_graph_name0,
    )

    # Add output gate
    dnn.add_node(
        in_nodes=[node_in, node_h],
        node_info={
            'name': output_gate,
            'type': 'output_gate Ot',
            'formula': 'Matmul(W_io,It)+Matmul(W_ho,H[t-1])+Bo',
            'Shape W_io': '[hidden_size, input_size]',
            'Shape W_ho': '[hidden_size, Proj_size]',
            'Shape out': ['batch', 'hidden_size']
        },
        graph_name=sub_graph_name0,
    )

    # Add cell update
    dnn.add_node(
        in_nodes=[node_in, node_h],
        node_info={
            'name': cell_update,
            'type': 'cell_update ct',
            'formula': 'tanh(Matmul(W_ic,Xt)+Matmul(W_hc,H[t-1])+Bc)',
            'Shape W_io': '[hidden_size, input_size]',
            'Shape W_ho': '[hidden_size, Proj_size]',
            'Shape out': ['batch', 'hidden_size']
        },
        graph_name=sub_graph_name0,
    )

    sub_graph_name1 = dnn.add_graph(graph_name+'_1', directed=True, subgraph=True, graph_attr={'style':'invis'})
    # Add cell state
    dnn.add_node(
        in_nodes=[node_c, forget_gate, input_gate, cell_update],
        node_info={
            'name': cell_state,
            'type': 'cell_state Ct',
            'formula': 'EleMul(Ft,C[t-1])+EleMul(It,ct)',
            'Shape out': ['batch', 'hidden_size']
        },
        graph_name=sub_graph_name1,
    )

    # Add hidden state
    dnn.add_node(
        in_nodes=[cell_state, output_gate],
        node_info={
            'name': hidden_state,
            'type': 'hidden_state Ht',
            'formula': 'EleMul(Ot, tanh(Ct))',
            'Shape out': ['batch', 'hidden_size']
        },
        graph_name=sub_graph_name1,
    )

    dnn.merge_subgraph(graph_name, sub_graph_name0)
    dnn.merge_subgraph(graph_name, sub_graph_name1)
    dnn.merge_subgraph(dnn.name, graph_name)
    return hidden_state, cell_state

# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#lstm
# https://d2l.ai/chapter_recurrent-modern/lstm.html
def LSTM(name, seq_len=1, num_layers=1):
    # Create main graph
    dnn = dnndraw.graph(name=name, layout="BT")

    it_cache = [None for _ in range(seq_len)]
    ht_cache = [None for _ in range(num_layers)]
    ct_cache = [None for _ in range(num_layers)]

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
        ct = dnn.add_node(
            in_nodes=[],
            node_info={
                'name': f'c_l{i}',
                'type': 'Initial_Cell',
                'shape': ['batch', 'hidden_size']
            }
        )
        ht_cache[i] = ht
        ct_cache[i] = ct
    # Add LSTM cell
    for i in range(num_layers):
        for j in range(seq_len):
            ht, ct = Add_LSTM_Cell(
                dnn,
                name=f'Cell_l{i}_t{j}',
                node_in=it_cache[j],
                node_h=ht_cache[i],
                node_c=ct_cache[i],
            )
            it_cache[j] = ht
            ht_cache[i] = ht
            ct_cache[i] = ct
    return dnn

if __name__ == '__main__':
    dnn = LSTM("LSTM CELL", seq_len=1, num_layers=1)
    print(dnn.source())
    dnn.export(format='svg')
