import dnndraw

# Facebook DLRM
# https://arxiv.org/abs/1705.07750

def Add_MLP_Arch(dnn, name, in_nodes, embedding_dims, out_activation='None'):
    layer_num = len(embedding_dims)
    name = dnn.add_graph(name, directed=True, subgraph=True)
    dnn.add_node(in_nodes = in_nodes,
                 node_info = {
                    'name': '{}_{}'.format(name, 1),
                    'type': 'Dense',
                    'activation': 'Relu',
                    'out_shape': ['batch', embedding_dims[0]]
                },
                graph_name=name)
    for i, embedding_dim in enumerate(embedding_dims[1:-1]):
        dnn.add_node(in_nodes = ['{}_{}'.format(name, i+1)],
                     node_info = {
                        'name': '{}_{}'.format(name, i+2),
                        'type': 'Dense',
                        'activation': 'Relu',
                        'out_shape': ['batch', embedding_dim]
                    },
                    graph_name=name)
    dnn.add_node(in_nodes = ['{}_{}'.format(name, layer_num-1)],
                 node_info = {
                    'name': '{}_{}'.format(name, layer_num),
                    'type': 'Dense',
                    'activation': out_activation,
                    'out_shape': ['batch', embedding_dims[-1]]
                },
                graph_name=name)
    dnn.merge_subgraph(dnn.name, name)
    return '{}_{}'.format(name, layer_num)

def Add_Embedding_Arch(dnn, name, in_nodes, embedding_sizes, embedding_dim):
    name = dnn.add_graph(name, directed=True, subgraph=True)
    out_name = []
    for i, embedding_size in enumerate(embedding_sizes):
        out_name.append('{}_{}'.format(name, embedding_size))
        dnn.add_node(in_nodes = in_nodes,
                     node_info = {
                        'name': '{}_{}'.format(name, embedding_size),
                        'type': 'Embedding',
                        'embedding_size': embedding_size,
                        'out_shape': ['batch', embedding_dim]
                    },
                    graph_name=name)
    dnn.merge_subgraph(dnn.name, name)
    return out_name

def Add_Interaction_Arch(dnn, name, in_nodes):
    name = dnn.add_graph(name, directed=True, subgraph=True)
    dnn.add_node(in_nodes=in_nodes, node_info={'name': name+'_concat', 'type': 'Concat', 'out_shape': ['batch', 27, 16]}, graph_name=name)
    dnn.add_node(in_nodes=[name+'_concat'], node_info={'name': name+'_tranpose', 'type': 'Tranpose', 'out_shape': ['batch', 16, 27]}, graph_name=name)
    dnn.add_node(in_nodes=[name+'_concat', name+'_tranpose'], node_info={'name': name+'_matmul', 'type': 'Matmul', 'out_shape': ['batch', 27, 27]}, graph_name=name)
    dnn.add_node(in_nodes=[name+'_matmul'], node_info={'name': name+'_mask', 'type': 'Mask', 'out_shape': ['batch', '27*26/2=351']}, graph_name=name)
    dnn.add_node(in_nodes=[in_nodes[0], name+'_mask'], node_info={'name': name+'_concat_top', 'type': 'Concat', 'out_shape': ['batch', 367]}, graph_name=name)
    dnn.merge_subgraph(dnn.name, name)
    return name+'_concat_top'

if __name__ == '__main__':
    dnn = dnndraw.graph(name="DLRM")

    # first layer
    dnn.add_node(in_nodes=[], node_info={'name': 'dense_input', 'type': 'Data', 'shape': ['batch', 'dense_feature_num=13']})
    dnn.add_node(in_nodes=[], node_info={'name': 'spare_input', 'type': 'Data', 'shape': ['spare_feature_num=26', 'batch']})

    mlp_bot_out_name = Add_MLP_Arch(dnn, 'MLP_Bottom', ['dense_input'], [512, 256, 64, 16], 'Sigmoid')

    emb_out_name = Add_Embedding_Arch(dnn, 'Embedding', ['spare_input'], ['c1', 'c2', 'c3', 'cx', 'c26'], embedding_dim='embedding_dim=16')

    iteract_name = Add_Interaction_Arch(dnn, 'Interaction', [mlp_bot_out_name, *emb_out_name])

    mlp_top_out_name = Add_MLP_Arch(dnn, 'MLP_Top', [iteract_name], [512, 256, 1], 'Sigmoid')

    dnn.save(file_path=dnn.name+'.pkl')
    dnn.show(format='png') # format: png, svg, pdf, ...

