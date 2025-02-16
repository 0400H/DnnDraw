import dnndraw


# DeepMind I3D
# https://arxiv.org/abs/1705.07750

def Add_InceptionModul3D(in_nodes, node_info):
    layer_name = node_info['name']
    layer_pad = node_info['padding']
    layer_channel = node_info['channel']
    layer_output_shape = node_info['output shape']

    dnn.add_node(in_nodes, {'name': layer_name+'_0a', 'type': 'Unit3D', 'layer_1': 'Pad', 'padding_1': layer_pad[0], 'layer_2': 'Conv3D', 'kernel': [1, 1, 1], 'stride': [1, 1, 1], 'padding_2': 'None', 'normal, relu': 'True', 'output shape': [layer_channel[0]] + layer_output_shape[1:]})

    dnn.add_node(in_nodes, {'name': layer_name+'_1a', 'type': 'Unit3D', 'layer_1': 'Pad', 'padding_1': layer_pad[1], 'layer_2': 'Conv3D', 'kernel': [1, 1, 1], 'stride': [1, 1, 1], 'padding_2': 'None', 'normal, relu': 'True', 'output shape': [layer_channel[1]] + layer_output_shape[1:]})
    dnn.add_node([layer_name+'_1a'], {'name': layer_name+'_1b', 'type': 'Unit3D', 'layer_1': 'Pad_1', 'padding': layer_pad[2], 'layer_2': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding_2': 'None', 'normal, relu': 'True', 'output shape': [layer_channel[2]] + layer_output_shape[1:]})

    dnn.add_node(in_nodes, {'name': layer_name+'_2a', 'type': 'Unit3D', 'layer_1': 'Pad', 'padding_1': layer_pad[3], 'layer_2': 'Conv3D', 'kernel': [1, 1, 1], 'stride': [1, 1, 1], 'padding_2': 'None', 'normal, relu': 'True', 'output shape': [layer_channel[3]] + layer_output_shape[1:]})
    dnn.add_node([layer_name+'_2a'], {'name': layer_name+'_2b', 'type': 'Unit3D', 'layer_1': 'Pad', 'padding_1': layer_pad[4], 'layer_2': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding_2': 'None', 'normal, relu': 'True', 'output shape': [layer_channel[4]] + layer_output_shape[1:]})

    dnn.add_node(in_nodes, {'name': layer_name+'_3a', 'type': 'MaxPool3d', 'layer_1': 'Pad', 'padding_1': layer_pad[5], 'layer_2': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding_2': 'None', 'output shape': [layer_channel[6]] + layer_output_shape[1:]})
    dnn.add_node([layer_name+'_3a'], {'name': layer_name+'_3b', 'type': 'Unit3D', 'layer_1': 'Pad', 'padding_1': layer_pad[6], 'layer_2': 'Conv3D', 'kernel': [1, 1, 1], 'stride': [1, 1, 1], 'padding_2': 'None', 'normal, relu': 'True', 'output shape': [layer_channel[5]] + layer_output_shape[1:]})

    dnn.add_node([layer_name+'_0a', layer_name+'_1b', layer_name+'_2b', layer_name+'_3b'], {'name': layer_name, 'type': 'concat', 'output shape': layer_output_shape})
    pass

def I3D(graph_name):
    global dnn
    dnn = dnndraw.graph(graph_name, layout="TB")

    dnn.add_node([], {'name': 'input', 'input shape': [3, 64, 224, 224], 'note': 'channel, depth, height, width'})
    dnn.add_node(['input'], {'name': 'Conv3d_1a_7x7', 'type': 'Unit3D', 'layer_1': 'Pad', 'padding_1': [5, 5, 5], 'layer_2': 'Conv3D', 'kernel': [7, 7, 7], 'stride': [2, 2, 2], 'normal, relu': 'True', 'padding_2': 'None', 'output shape': [64, 32, 112, 112]})
    dnn.add_node(['Conv3d_1a_7x7'], {'name': 'MaxPool3d_2a_3x3', 'type': 'MaxPool3d', 'layer_1': 'Pad', 'padding_1': [0, 1, 1], 'layer_2': 'Conv3D', 'kernel': [1, 3, 3], 'stride': [1, 2, 2], 'padding_2': 'None', 'output shape': [64, 32, 56, 56]})
    dnn.add_node(['MaxPool3d_2a_3x3'], {'name': 'Conv3d_2b_1x1', 'type': 'Unit3D', 'layer_1': 'Pad', 'padding_1': [0, 0, 0], 'layer_2': 'Conv3D', 'kernel': [1, 1, 1], 'stride': [1, 1, 1], 'padding_2': 'None', 'normal, relu': 'True', 'output shape': [64, 32, 56, 56]})
    dnn.add_node(['Conv3d_2b_1x1'], {'name': 'Conv3d_2c_3x3', 'type': 'Unit3D', 'layer_1': 'Pad', 'padding_1': [2, 2, 2], 'layer_2': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding_2': 'None', 'normal, relu': 'True', 'output shape': [192, 32, 56, 56]})
    dnn.add_node(['Conv3d_2c_3x3'], {'name': 'MaxPool3d_3a_3x3', 'type': 'MaxPool3d', 'layer_1': 'Pad', 'padding_1': [0, 1, 1], 'layer_2': 'Conv3D', 'kernel': [1, 3, 3], 'stride': [1, 2, 2], 'padding_2': 'None', 'output shape': [192, 32, 28, 28]})

    Add_InceptionModul3D(['MaxPool3d_3a_3x3'], {'name': 'Mixed_3b', 'channel': [64,96,128,16,32,32,192], 'padding': [
            [0, 0, 0],
            [0, 0, 0], [2, 2, 2],
            [0, 0, 0], [2, 2, 2],
            [2, 2, 2], [0, 0, 0],
        ], 'output shape': [256, 32, 28, 28]})
    Add_InceptionModul3D(['Mixed_3b'], {'name': 'Mixed_3c', 'channel': [128,128,192,32,96,64,256], 'padding': [
            [0, 0, 0],
            [0, 0, 0], [2, 2, 2],
            [0, 0, 0], [2, 2, 2],
            [2, 2, 2], [0, 0, 0],
        ], 'output shape': [480, 32, 28, 28]})

    dnn.add_node(['Mixed_3c'], {'name': 'MaxPool3d_4a_3x3', 'type': 'MaxPool3d', 'layer_1': 'Pad', 'padding_1': [1, 1, 1], 'layer_2': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [2, 2, 2], 'padding_2': 'None', 'output shape': [480, 16, 14, 14]})

    Add_InceptionModul3D(['MaxPool3d_4a_3x3'], {'name': 'Mixed_4b', 'channel': [192,96,208,16,48,64,480], 'padding': [
            [0, 0, 0],
            [0, 0, 0], [2, 2, 2],
            [0, 0, 0], [2, 2, 2],
            [2, 2, 2], [0, 0, 0],
        ], 'output shape': [512, 16, 14, 14]})
    Add_InceptionModul3D(['Mixed_4b'], {'name': 'Mixed_4c', 'channel': [160,112,224,24,64,64,512], 'padding': [
            [0, 0, 0],
            [0, 0, 0], [2, 2, 2],
            [0, 0, 0], [2, 2, 2],
            [2, 2, 2], [0, 0, 0],
        ], 'output shape': [512, 16, 14, 14]})
    Add_InceptionModul3D(['Mixed_4c'], {'name': 'Mixed_4d', 'channel': [128,128,256,24,64,64,512], 'padding': [
            [0, 0, 0],
            [0, 0, 0], [2, 2, 2],
            [0, 0, 0], [2, 2, 2],
            [2, 2, 2], [0, 0, 0],
        ], 'output shape': [512, 16, 14, 14]})
    Add_InceptionModul3D(['Mixed_4d'], {'name': 'Mixed_4e', 'channel': [112,144,288,32,64,64,512], 'padding': [
            [0, 0, 0],
            [0, 0, 0], [2, 2, 2],
            [0, 0, 0], [2, 2, 2],
            [2, 2, 2], [0, 0, 0],
        ], 'output shape': [528, 16, 14, 14]})
    Add_InceptionModul3D(['Mixed_4e'], {'name': 'Mixed_4f', 'channel': [256,160,320,32,128,128,528], 'padding': [
            [0, 0, 0],
            [0, 0, 0], [2, 2, 2],
            [0, 0, 0], [2, 2, 2],
            [2, 2, 2], [0, 0, 0],
        ], 'output shape': [832, 16, 14, 14]})

    dnn.add_node(['Mixed_4f'], {'name': 'MaxPool3d_5a_2x2', 'type': 'MaxPool3d', 'layer_1': 'Pad', 'padding_1': [0, 0, 0], 'layer_2': 'Conv3D', 'kernel': [2, 2, 2], 'stride': [2, 2, 2], 'padding_2': 'None', 'output shape': [832, 8, 7, 7]})

    Add_InceptionModul3D(['MaxPool3d_5a_2x2'], {'name': 'Mixed_5b', 'channel': [256,160,320,32,128,128,832], 'padding': [
            [0, 0, 0],
            [0, 0, 0], [2, 2, 2],
            [0, 0, 0], [2, 2, 2],
            [2, 2, 2], [0, 0, 0],
        ], 'output shape': [832, 8, 7, 7]})
    Add_InceptionModul3D(['Mixed_5b'], {'name': 'Mixed_5c', 'channel': [384,192,384,48,128,128,832], 'padding': [
            [0, 0, 0],
            [0, 0, 0], [2, 2, 2],
            [0, 0, 0], [2, 2, 2],
            [2, 2, 2], [0, 0, 0],
        ], 'output shape': [1024, 8, 7, 7]})

    dnn.add_node(['Mixed_5c'], {'name': 'AvgPool', 'type': 'AvgPool3D', 'layer_1': 'Pad', 'padding_1': [0, 0, 0], 'layer_2': 'Conv3D', 'kernel': [2, 7, 7], 'stride': [1, 1, 1], 'padding_2': 'None', 'output shape': [1024, 7, 1, 1]})
    dnn.add_node(['AvgPool'], {'name': 'Dropout', 'type': 'Dropout', 'output shape': [1024, 7, 1, 1]})
    dnn.add_node(['Dropout'], {'name': 'Logits', 'type': 'Unit3D', 'layer_1': 'Pad', 'padding_1': [0, 0, 0], 'layer_2': 'Conv3D', 'kernel': [1, 1, 1], 'stride': [1, 1, 1], 'padding_2': 'None', 'normal, relu': 'False', 'output shape': [101, 7, 1, 1]})
    return dnn


if __name__ == '__main__':
    dnn = I3D('I3D_Topology')
    print(dnn.source())
    dnn.export(format='svg') # format: png, svg, pdf, ...
    # dnn.show()
