import dnndraw

dnn = dnndraw.graph(name="tinydnn", layout="TB")

# first layer
dnn.add_node(in_nodes=[], node_info={'name': 'layer_1', 'Type': 'Conv3D', 'kernel': [1, 1, 1], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

dnn.add_node(in_nodes=['layer_1'], node_info={'name': 'layer_2', 'Type': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

dnn.add_node(in_nodes=['layer_1'], node_info={'name': 'layer_3', 'Type': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

# end layer
dnn.add_node(in_nodes=['layer_2', 'layer_3'], node_info={'name': 'layer_4', 'Type': 'Concat'})

print(dnn.source())
dnn.export(format='png') # format: png, svg, pdf, ...
# dnn.show()
