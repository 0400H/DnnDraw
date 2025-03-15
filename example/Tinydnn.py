import dnndraw

dnn = dnndraw.graph(name="Tinydnn", layout="TB")

# first layer
dnn.add_node(
    in_nodes=[],
    node_info={
        'name': 'input',
        'operator': 'Data',
    })

dnn.add_node(
    in_nodes={"input": "[batch, h, w, c]"},
    node_info={
        'name': 'layer_1',
        'operator': 'Conv3D',
        'kernel': [1, 1, 1],
        'stride': [1, 1, 1],
        'padding': 'none',
        'normal, relu': 'True'
    })

dnn.add_node(
    in_nodes=['layer_1'],
    node_info={
        'name': 'layer_2',
        'operator': 'Conv3D',
        'kernel': [3, 3, 3],
        'stride': [1, 1, 1],
        'padding': 'none',
        'normal, relu': 'True'
    })

dnn.add_node(
    in_nodes=['layer_1'],
    node_info={
        'name': 'layer_3',
        'operator': 'Conv3D',
        'kernel': [3, 3, 3],
        'stride': [1, 1, 1],
        'padding': 'none',
        'normal, relu': 'True'
    })

# end layer
dnn.add_node(
    in_nodes={
        'layer_2': "",
        'layer_3': "",
    },
    node_info={
        'name': 'layer_4',
        'operator': 'Concat'
    })

print(dnn.source())
dnn.export(format='svg') # format: png, svg, pdf, ...
# dnn.show()
