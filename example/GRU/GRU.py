import dnndraw


def GRU(name, seq_len=1, num_layers=1):
    # Create main graph
    dnn = dnndraw.graph(name=name, layout="LR")

    it_cache = [None for _ in range(seq_len)]
    ht_cache = [None for _ in range(num_layers)]

    # initial input data
    g_data = dnn.create_graph('data', directed=True, attr={'style':'invis', 'rank':'same'})
    for i in range(seq_len):
        it = dnn.add_node(
            in_nodes=[],
            node_info={
                'name': f'In_T{i}',
                'type': 'Input data',
                'shape': ['batch', 'input_size']
            },
            node_attr=dnn.node_colors[0],
            graph=g_data,
        )
        it_cache[i] = it
    dnn.merge_graph(g_data)

    # Add initial states
    g_h = dnn.create_graph('states', directed=True, attr={'style':'invis'})
    for i in range(num_layers):
        ht = dnn.add_node(
            in_nodes=[],
            node_info={
                'name': f'H_L{i}',
                'type': 'Initial Hidden',
                'shape': ['batch', 'hidden_size']
            },
            node_attr=dnn.node_colors[1],
            graph=g_h,
        )
        ht_cache[i] = ht
    dnn.merge_graph(g_h)

    It = "It"
    # Add GRU cell
    for i in range(num_layers):
        if i:
            It = "Ht"
        graph_seq = dnn.create_graph(dnn.name+f'_seq_{i}', directed=True, attr={'style':'invis'})
        for j in range(seq_len):
            name = dnn.add_node(
                in_nodes={
                    ht_cache[i]: "Ht",
                    it_cache[j]: It,
                },
                node_info={
                    'name': f'Cell_L{i}T{i+j}',
                    'type': 'GRU Cell',
                    'shape Ht': ['batch', 'hidden_size'],
                },
                graph=graph_seq,
            )
            it_cache[j] = name
            ht_cache[i] = name
        dnn.merge_graph(graph_seq)
    return dnn

if __name__ == '__main__':
    dnn = GRU("GRU", seq_len=3, num_layers=3)
    print(dnn.source())
    dnn.export(format='svg')

