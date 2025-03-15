import dnndraw

# Facebook DLRM
# https://arxiv.org/abs/1705.07750

def Add_MLP_Arch(dnn, name, in_nodes, embedding_dims, out_activation='none'):
    layer_num = len(embedding_dims)
    cg, g = dnn.create_graph(name, directed=True, has_border=True)
    dnn.add_node(in_nodes = in_nodes,
                 node_info = {
                    'name': '{}_{}'.format(name, 1),
                    'operator': 'Dense',
                    'activation': 'Relu',
                    'out_shape': ['batch', embedding_dims[0]]
                },
                graph=g)
    for i, embedding_dim in enumerate(embedding_dims[1:-1]):
        dnn.add_node(in_nodes = ['{}_{}'.format(name, i+1)],
                     node_info = {
                        'name': '{}_{}'.format(name, i+2),
                        'operator': 'Dense',
                        'activation': 'Relu',
                        'out_shape': ['batch', embedding_dim]
                    },
                    graph=g)
    dnn.add_node(in_nodes = ['{}_{}'.format(name, layer_num-1)],
                 node_info = {
                    'name': '{}_{}'.format(name, layer_num),
                    'operator': 'Dense',
                    'activation': out_activation,
                    'out_shape': ['batch', embedding_dims[-1]]
                },
                graph=g)
    dnn.merge_graph(g, cg)
    dnn.merge_graph(cg)
    return '{}_{}'.format(name, layer_num)

def Add_Embedding_Arch(dnn, name, in_nodes, embedding_sizes, embedding_dim):
    cg, g = dnn.create_graph(name, directed=True, has_border=True)
    out_name = []
    for i, embedding_size in enumerate(embedding_sizes):
        out_name.append('{}_{}'.format(name, embedding_size))
        dnn.add_node(in_nodes = in_nodes,
                     node_info = {
                        'name': '{}_{}'.format(name, embedding_size),
                        'operator': 'Embedding',
                        'embedding_size': embedding_size,
                        'out_shape': ['batch', embedding_dim]
                    },
                    graph=g)
    dnn.merge_graph(g, cg)
    dnn.merge_graph(cg)
    return out_name

def Add_Interaction_Arch(dnn, name, in_nodes):
    cg, g = dnn.create_graph(name, directed=True, has_border=True)
    dnn.add_node(in_nodes=in_nodes, node_info={'name': name+'_concat', 'operator': 'Concat', 'out_shape': ['batch', 27, 16]}, graph=g)
    dnn.add_node(in_nodes=[name+'_concat'], node_info={'name': name+'_tranpose', 'operator': 'Tranpose', 'out_shape': ['batch', 16, 27]}, graph=g)
    dnn.add_node(in_nodes=[name+'_concat', name+'_tranpose'], node_info={'name': name+'_matmul', 'operator': 'Matmul', 'out_shape': ['batch', 27, 27]}, graph=g)
    dnn.add_node(in_nodes=[name+'_matmul'], node_info={'name': name+'_mask', 'operator': 'Mask', 'out_shape': ['batch', '27*26/2=351']}, graph=g)
    dnn.add_node(in_nodes=[in_nodes[0], name+'_mask'], node_info={'name': name+'_concat_top', 'operator': 'Concat', 'out_shape': ['batch', 367]}, graph=g)
    dnn.merge_graph(g, cg)
    dnn.merge_graph(cg)
    return name+'_concat_top'

if __name__ == '__main__':
    dnn = dnndraw.graph(name="DLRM", layout="BT")

    # first layer
    dnn.add_node(in_nodes=[], node_info={'name': 'dense_input', 'operator': 'Data', 'shape': ['batch', 'dense_feature_num=13']})
    dnn.add_node(in_nodes=[], node_info={'name': 'spare_input', 'operator': 'Data', 'shape': ['spare_feature_num=26', 'batch']})

    mlp_bot_out_name = Add_MLP_Arch(dnn, 'MLP_Bottom', ['dense_input'], [512, 256, 64, 16], 'Sigmoid')

    emb_out_name = Add_Embedding_Arch(dnn, 'Embedding', ['spare_input'], ['c1', 'c2', 'c3', 'cx', 'c26'], embedding_dim='embedding_dim=16')

    iteract_name = Add_Interaction_Arch(dnn, 'Interaction', [mlp_bot_out_name, *emb_out_name])

    mlp_top_out_name = Add_MLP_Arch(dnn, 'MLP_Top', [iteract_name], [512, 256, 1], 'Sigmoid')

    print(dnn.source())
    dnn.export(format='svg') # format: png, svg, pdf, ...
    # dnn.show()
