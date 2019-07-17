import os, sys
__Father_Root__ = os.path.dirname(os.path.abspath(__file__)) + '/'
__Project_Root__ = os.path.dirname(__Father_Root__ + '../')
sys.path.append(__Project_Root__)

from DnnDraw import DnnDraw

# DeepMind I3D
# https://arxiv.org/abs/1705.07750

def Add_InceptionModul3D(dnn_class, in_name, info_dict):
    layer_name = info_dict['name']
    layer_pad = info_dict['padding']
    layer_param = info_dict['param']
    layer_output_shape = info_dict['output shape']
    layer_channel = [int(dnn_class[0].info_list[-1]['output shape'][1:-1].split(', ')[0])]

    dnn_class[0].add_layer(in_name, {'name': layer_name+'_0a', 'type': 'Unit3D', 'kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'padding': layer_pad[0], 'normal, relu': 'TRUE', 'output shape': str([layer_param[0]] + layer_output_shape[1:])})

    dnn_class[0].add_layer(in_name, {'name': layer_name+'_1a', 'type': 'Unit3D', 'kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'padding': layer_pad[1], 'normal, relu': 'TRUE', 'output shape': str([layer_param[1]] + layer_output_shape[1:])})
    dnn_class[0].add_layer([layer_name+'_1a'], {'name': layer_name+'_1b', 'type': 'Unit3D', 'kernel': r'[3, 3, 3]', 'stride': r'[1, 1, 1]', 'padding': layer_pad[2], 'normal, relu': 'TRUE', 'output shape': str([layer_param[2]] + layer_output_shape[1:])})

    dnn_class[0].add_layer(in_name, {'name': layer_name+'_2a', 'type': 'Unit3D', 'kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'padding': layer_pad[3], 'normal, relu': 'TRUE', 'output shape': str([layer_param[3]] + layer_output_shape[1:])})
    dnn_class[0].add_layer([layer_name+'_2a'], {'name': layer_name+'_2b', 'type': 'Unit3D', 'kernel': r'[3, 3, 3]', 'stride': r'[1, 1, 1]', 'padding': layer_pad[4], 'normal, relu': 'TRUE', 'output shape': str([layer_param[4]] + layer_output_shape[1:])})

    dnn_class[0].add_layer(in_name, {'name': layer_name+'_3a', 'type': 'MaxPool3dSamePadding', 'kernel': r'[3, 3, 3]', 'stride': r'[1, 1, 1]', 'padding': layer_pad[5], 'output shape': str(layer_channel + layer_output_shape[1:])})
    dnn_class[0].add_layer([layer_name+'_3a'], {'name': layer_name+'_3b', 'type': 'Unit3D', 'kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'padding': layer_pad[6], 'normal, relu': 'TRUE', 'output shape': str([layer_param[5]] + layer_output_shape[1:])})

    dnn_class[0].add_layer([layer_name+'_0a', layer_name+'_1b', layer_name+'_2b', layer_name+'_3b'], {'name': layer_name, 'type': 'concat', 'output shape': str(layer_output_shape)})

def I3D (project):
    padding = [
        r'[5, 5, 5]',
        r'[0, 1, 1]',
        r'[0, 0, 0]',
        r'[2, 2, 2]',
        r'[0, 1, 1]',
        [
            r'[0, 0, 0]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[2, 2, 2]', r'[0, 0, 0]',
        ],
        [
            r'[0, 0, 0]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[2, 2, 2]', r'[0, 0, 0]',
        ],
        r'[1, 1, 1]',
        [
            r'[0, 0, 0]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[2, 2, 2]', r'[0, 0, 0]',
        ],
        [
            r'[0, 0, 0]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[2, 2, 2]', r'[0, 0, 0]',
        ],
        [
            r'[0, 0, 0]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[2, 2, 2]', r'[0, 0, 0]',
        ],
        [
            r'[0, 0, 0]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[2, 2, 2]', r'[0, 0, 0]',
        ],
        [
            r'[0, 0, 0]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[2, 2, 2]', r'[0, 0, 0]',
        ],
        r'[0, 0, 0]',
        [
            r'[0, 0, 0]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[2, 2, 2]', r'[0, 0, 0]',
        ],
        [
            r'[0, 0, 0]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[0, 0, 0]', r'[2, 2, 2]',
            r'[2, 2, 2]', r'[0, 0, 0]',
        ],
        r'[0, 0, 0]',
        r'[0, 0, 0]',
    ]

    nn = DnnDraw(project, '100,100', 'svg')

    nn.add_layer([], {'name': 'input', 'input shape': r'[3, 64, 224, 224]'})
    nn.add_layer(['input'], {'name': 'Conv3d_1a_7x7', 'type': 'Unit3D', 'kernel': r'[7, 7, 7]', 'stride': r'[2, 2, 2]', 'padding': padding[0], 'normal, relu': 'TRUE', 'output shape': r'[64, 32, 112, 112]'})
    nn.add_layer(['Conv3d_1a_7x7'], {'name': 'MaxPool3d_2a_3x3', 'type': 'MaxPool3dSamePadding', 'kernel': r'[1, 3, 3]', 'stride': r'[1, 2, 2]', 'padding': padding[1], 'output shape': r'[64, 32, 56, 56]'})
    nn.add_layer(['MaxPool3d_2a_3x3'], {'name': 'Conv3d_2b_1x1', 'type': 'Unit3D','kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'padding': padding[2], 'normal, relu': 'TRUE', 'output shape': r'[64, 32, 56, 56]'})
    nn.add_layer(['Conv3d_2b_1x1'], {'name': 'Conv3d_2c_3x3', 'type': 'Unit3D', 'kernel': r'[3, 3, 3]', 'stride': r'[1, 1, 1]', 'padding': padding[3], 'normal, relu': 'TRUE', 'output shape': r'[192, 32, 56, 56]'})
    nn.add_layer(['Conv3d_2c_3x3'], {'name': 'MaxPool3d_3a_3x3', 'type': 'MaxPool3dSamePadding', 'kernel': r'[1, 3, 3]', 'stride': r'[1, 2, 2]', 'padding': padding[4], 'output shape': r'[192, 32, 28, 28]'})
    Add_InceptionModul3D([nn], ['MaxPool3d_3a_3x3'], {'name': 'Mixed_3b', 'param': [64,96,128,16,32,32], 'padding': padding[5], 'output shape': [256, 32, 28, 28]})
    Add_InceptionModul3D([nn], ['Mixed_3b'], {'name': 'Mixed_3c', 'param': [128,128,192,32,96,64], 'padding': padding[6], 'output shape': [480, 32, 28, 28]})
    nn.add_layer(['Mixed_3c'], {'name': 'MaxPool3d_4a_3x3', 'type': 'MaxPool3dSamePadding', 'kernel': r'[3, 3, 3]', 'stride': r'[2, 2, 2]', 'padding': padding[7], 'output shape': r'[480, 16, 14, 14]'})
    Add_InceptionModul3D([nn], ['MaxPool3d_4a_3x3'], {'name': 'Mixed_4b', 'param': [192,96,208,16,48,64], 'padding': padding[8], 'output shape': [512, 16, 14, 14]})
    Add_InceptionModul3D([nn], ['Mixed_4b'], {'name': 'Mixed_4c', 'param': [160,112,224,24,64,64], 'padding': padding[9], 'output shape': [512, 16, 14, 14]})
    Add_InceptionModul3D([nn], ['Mixed_4c'], {'name': 'Mixed_4d', 'param': [128,128,256,24,64,64], 'padding': padding[10], 'output shape': [512, 16, 14, 14]})
    Add_InceptionModul3D([nn], ['Mixed_4d'], {'name': 'Mixed_4e', 'param': [112,144,288,32,64,64], 'padding': padding[11], 'output shape': [528, 16, 14, 14]})
    Add_InceptionModul3D([nn], ['Mixed_4e'], {'name': 'Mixed_4f', 'param': [256,160,320,32,128,128], 'padding': padding[12], 'output shape': [832, 16, 14, 14]})
    nn.add_layer(['Mixed_4f'], {'name': 'MaxPool3d_5a_2x2', 'type': 'MaxPool3dSamePadding', 'kernel': r'[2, 2, 2]', 'stride': r'[2, 2, 2]', 'padding': padding[13], 'output shape': r'[832, 8, 7, 7]'})
    Add_InceptionModul3D([nn], ['MaxPool3d_5a_2x2'], {'name': 'Mixed_5b', 'param': [256,160,320,32,128,128], 'padding': padding[14], 'output shape': [832, 8, 7, 7]})
    Add_InceptionModul3D([nn], ['Mixed_5b'], {'name': 'Mixed_5c', 'param': [384,192,384,48,128,128], 'padding': padding[15], 'output shape': [1024, 8, 7, 7]})
    nn.add_layer(['Mixed_5c'], {'name': 'AvgPool', 'type': 'AvgPool3D', 'kernel': r'[2, 7, 7]', 'stride': r'[1, 1, 1]', 'padding': padding[16], 'output shape': r'[1024, 7, 1, 1]'})
    nn.add_layer(['AvgPool'], {'name': 'Dropout', 'type': 'Dropout', 'output shape': r'[1024, 7, 1, 1]'})
    nn.add_layer(['Dropout'], {'name': 'Logits', 'type': 'Unit3D', 'kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'padding': padding[17], 'normal, relu': 'False', 'output shape': r'[101, 7, 1, 1]'})

    nn.show()
    nn.save('png')
    return None


if __name__ == '__main__':
    I3D('I3D_Topology')