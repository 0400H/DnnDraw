import dnndraw


def LSTM(name, seq_len=1, num_layers=1):
    # Create main graph
    dnn = dnndraw.graph(name=name, layout="LR")

    it_cache = [None for _ in range(seq_len)]
    ht_cache = [None for _ in range(num_layers)]
    ct_cache = [None for _ in range(num_layers)]

    # initial input data
    for i in range(seq_len):
        it = dnn.add_node(
            in_nodes=[],
            node_info={
                'name': f'In_L0T{i}',
                'type': 'Input data',
                'shape': ['batch', 'input_size']
            },
        )
        it_cache[i] = it

    # Add initial states
    for i in range(num_layers):
        ht = dnn.add_node(
            in_nodes=[],
            node_info={
                'name': f'H_L{i}T0',
                'type': 'Initial Hidden',
                'shape': ['batch', 'Proj_size']
            }
        )
        ct = dnn.add_node(
            in_nodes=[],
            node_info={
                'name': f'C_L{i}T0',
                'type': 'Initial Cell',
                'shape': ['batch', 'hidden_size']
            }
        )
        ht_cache[i] = ht
        ct_cache[i] = ct

    It = "It"
    # Add LSTM cell
    for i in range(num_layers):
        if i:
            It = "Ht"
        graph_seq = dnn.add_graph(dnn.name+f'_seq_{i}', directed=True, subgraph=True, graph_attr={'style':'invis'})
        for j in range(seq_len):
            Ct = "Ct"
            if j:
                Ct = "Ct, Ht"
            name = dnn.add_node(
                in_nodes={
                    it_cache[j]: It,
                    ht_cache[i]: "Ht",
                    ct_cache[i]: Ct,
                },
                node_info={
                    'name': f'Cell_L{i}T{i+j}',
                    'type': 'LSTM Cell',
                    'shape Ht': ['batch', 'Proj_size'],
                    'shape Ct': ['batch', 'hidden_size'],
                },
                graph_name=graph_seq,
            )
            it_cache[j] = name
            ht_cache[i] = name
            ct_cache[i] = name
        dnn.merge_subgraph(dnn.name, graph_seq)
    return dnn

if __name__ == '__main__':
    dnn = LSTM("LSTM", seq_len=3, num_layers=3)
    print(dnn.source())
    dnn.export(format='svg')

