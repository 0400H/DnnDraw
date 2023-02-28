import os, sys
__Father_Root__ = os.path.dirname(os.path.abspath(__file__)) + '/'
__Project_Root__ = os.path.dirname(__Father_Root__ + '../')
sys.path.append(__Project_Root__)
import dnndraw


dnn = dnndraw.graph(name="tinydnn")

# first layer
dnn.add_node(in_nodes=[], node_info={'name': 'layer_1', 'Type': 'Conv3D', 'kernel': [1, 1, 1], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

dnn.add_node(in_nodes=['layer_1'], node_info={'name': 'layer_2', 'Type': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

dnn.add_node(in_nodes=['layer_1'], node_info={'name': 'layer_3', 'Type': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

# end layer
dnn.add_node(in_nodes=['layer_2', 'layer_3'], node_info={'name': 'layer_4', 'Type': 'Concat'})

dnn.print()
dnn.save(format='png', file_path=dnn.name+'.json') # format: png, svg, pdf, ...
dnn.show()
