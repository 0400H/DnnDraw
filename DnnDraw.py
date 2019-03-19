# -*- coding: UTF-8 -*-

import graphviz as gz

class GraphViz (object):
    def __init__(self):
        self._node = set()
        self.graph = dict()
        return None

    def view(self, graph_name):
        self.graph[graph_name].view()
        return None

    def render(self, graph_name, out_format):
        self.graph[graph_name].format = out_format
        self.graph[graph_name].render()
        return None

    def save_resource(self, graph_name, file_path):
        self.graph[graph_name].write(file_path)
        return None

    def save_picture(self, graph_name, file_path):
        self.graph[graph_name].draw(file_path)
        return None

    def add_graph(self, graph_name, directed=False, graph_size='8,5', node_color='lightblue2', graph_style='filled', out_format='pdf'):
        if graph_name in self.graph:
            print('graph ', graph_name, ' already exists!')
            return False
        if directed == False:
            self.graph[graph_name] = gz.Graph(graph_name, format=out_format)
        else:
            self.graph[graph_name] = gz.Digraph(graph_name, format=out_format)
        if node_color != None:
            self.graph[graph_name].node_attr.update(color=node_color, shape= 'record', style=graph_style)
        self.graph[graph_name].attr(size=graph_size)
        return True

    # graphviz some shape: 'circle', 'box', 'record'
    # https://www.graphviz.org/doc/info/shapes.html
    def add_node(self, graph_name, node_in, name, info, node_shape='box', font_size='8'):
        if name in self._node:
            print('node ', name, ' already exists!')
            return False

        self._node.add(name)
        self.graph[graph_name].node(name, label=info, shape=node_shape, fontsize=font_size)

        if node_in != []:
            for in_layer in node_in:
                self.graph[graph_name].edge(in_layer, name)

        return True

pass

class DNN (GraphViz):
    def __init__(self, name, size, out_format='svg'):
        GraphViz.__init__(self)
        self._dnn_name = name
        self.add_graph(self._dnn_name, True, size, 'lightblue2', 'filled', out_format)
        return None

    def info_format(self, info_dict):
        key_info = r'{'
        value_info = r'{'
        length = len(info_dict)
        for key in info_dict:
            key_info = key_info + key + '|'
            value_info = value_info + info_dict[key] + '|'
        info = key_info[:-1] + r'} | ' + value_info[:-1] + r'}'
        return info

    def add_layer(self, layer_in_name, info_dict):
        info = self.info_format(info_dict)
        layer_name = info_dict['name']
        return self.add_node(self._dnn_name, layer_in_name, layer_name, info, 'record')

    def show(self):
        return self.view(self._dnn_name)

    def save(self, out_format):
        return self.render(self._dnn_name, out_format)

pass

def Add_InceptionModul3D(dnn_class, in_name, info_list):
    layer_name = info_list['name']
    layer_type = info_list['type']

    param = str(info_list['param'])[1:-1]
    param = param.split(',', param.count(','))

    in_channel = param[0]
    out_channel = str(int(param[1]) + int(param[3]) + int(param[5]) + int(param[6]))

    dnn_class[0].add_layer(in_name, {'name': layer_name+'_0a', 'type': 'Unit3D', 'channel': r'[%s, %s]'%(in_channel, param[1]), 'kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'normal, relu': 'True'})

    dnn_class[0].add_layer(in_name, {'name': layer_name+'_1a', 'type': 'Unit3D', 'channel': r'[%s, %s]'%(in_channel, param[2]), 'kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'normal, relu': 'True'})
    dnn_class[0].add_layer([layer_name+'_1a'], {'name': layer_name+'_1b', 'type': 'Unit3D', 'channel': r'[%s, %s]'%(param[2], param[3]), 'kernel': r'[3, 3, 3]', 'stride': r'[1, 1, 1]', 'normal, relu': 'True'})

    dnn_class[0].add_layer(in_name, {'name': layer_name+'_2a', 'type': 'Unit3D', 'channel': r'[%s, %s]'%(in_channel, param[4]), 'kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'normal, relu': 'True'})
    dnn_class[0].add_layer([layer_name+'_2a'], {'name': layer_name+'_2b', 'type': 'Unit3D', 'channel': r'[%s, %s]'%(param[4], param[5]), 'kernel': r'[3, 3, 3]', 'stride': r'[1, 1, 1]', 'normal, relu': 'True'})

    dnn_class[0].add_layer(in_name, {'name': layer_name+'_3a', 'type': 'MaxPool3dSamePadding', 'kernel': r'[3, 3, 3]', 'stride': r'[1, 1, 1]'})
    dnn_class[0].add_layer([layer_name+'_3a'], {'name': layer_name+'_3b', 'type': 'Unit3D', 'channel': r'[%s, %s]'%(in_channel, param[6]), 'kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'normal, relu': 'True'})

    dnn_class[0].add_layer([layer_name+'_0a', layer_name+'_1b', layer_name+'_2b', layer_name+'_3b'], {'name': layer_name, 'type': 'concat', 'channel': out_channel})

if __name__ == '__main__':
    project = 'I3D_Pology'
    nn = DNN(project, '50,100', 'svg')

    nn.add_layer([], {'name': 'input', 'shape': r'[32, 3, 64, 224, 224]'})
    nn.add_layer(['input'], {'name': 'Conv3d_1a_7x7', 'type': 'Unit3D', 'channel': r'[3, 64]', 'kernel': r'[7, 7, 7]', 'stride': r'[2, 2, 2]', 'padding': r'[3, 3, 3]', 'normal, relu': 'True'})
    nn.add_layer(['Conv3d_1a_7x7'], {'name': 'MaxPool3d_2a_3x3', 'type': 'MaxPool3dSamePadding', 'kernel': r'[1, 3, 3]', 'stride': r'[1, 2, 2]'})
    nn.add_layer(['MaxPool3d_2a_3x3'], {'name': 'Conv3d_2b_1x1', 'type': 'Unit3D', 'channel': r'[64, 64]', 'kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'padding': r'0', 'normal, relu': 'True'})
    nn.add_layer(['Conv3d_2b_1x1'], {'name': 'Conv3d_2c_3x3', 'type': 'Unit3D', 'channel': r'[64, 192]', 'kernel': r'[3, 3, 3]', 'stride': r'[1, 1, 1]', 'padding': r'1', 'normal, relu': 'True'})
    nn.add_layer(['Conv3d_2c_3x3'], {'name': 'MaxPool3d_3a_3x3', 'type': 'MaxPool3dSamePadding', 'kernel': r'[1, 3, 3]', 'stride': r'[1, 2, 2]'})
    Add_InceptionModul3D([nn], ['MaxPool3d_3a_3x3'], {'name': 'Mixed_3b', 'type': 'InceptionModule3D', 'param': r'[192,64,96,128,16,32,32]'})
    Add_InceptionModul3D([nn], ['Mixed_3b'], {'name': 'Mixed_3c', 'type': 'InceptionModule3D', 'param': r'[256,128,128,192,32,96,64]'})
    nn.add_layer(['Mixed_3c'], {'name': 'MaxPool3d_4a_3x3', 'type': 'MaxPool3dSamePadding', 'kernel': r'[3, 3, 3]', 'stride': r'[2, 2, 2]'})
    Add_InceptionModul3D([nn], ['MaxPool3d_4a_3x3'], {'name': 'Mixed_4b', 'type': 'InceptionModule3D', 'param': r'['+str(128+192+96+64)+r',192,96,208,16,48,64]'})
    Add_InceptionModul3D([nn], ['Mixed_4b'], {'name': 'Mixed_4c', 'type': 'InceptionModule3D', 'param': r'['+str(192+208+48+64)+r',160,112,224,24,64,64]'})
    Add_InceptionModul3D([nn], ['Mixed_4c'], {'name': 'Mixed_4d', 'type': 'InceptionModule3D', 'param': r'['+str(160+224+64+64)+r',128,128,256,24,64,64]'})
    Add_InceptionModul3D([nn], ['Mixed_4d'], {'name': 'Mixed_4e', 'type': 'InceptionModule3D', 'param': r'['+str(128+256+64+64)+r',112,144,288,32,64,64]'})
    Add_InceptionModul3D([nn], ['Mixed_4e'], {'name': 'Mixed_4f', 'type': 'InceptionModule3D', 'param': r'['+str(112+288+64+64)+r',256,160,320,32,128,128]'})
    nn.add_layer(['Mixed_4f'], {'name': 'MaxPool3d_5a_2x2', 'type': 'MaxPool3dSamePadding', 'kernel': r'[2, 2, 2]', 'stride': r'[2, 2, 2]'})
    Add_InceptionModul3D([nn], ['MaxPool3d_5a_2x2'], {'name': 'Mixed_5b', 'type': 'InceptionModule3D', 'param': r'['+str(256+320+128+128)+r',256,160,320,32,128,128]'})
    Add_InceptionModul3D([nn], ['Mixed_5b'], {'name': 'Mixed_5c', 'type': 'InceptionModule3D', 'param': r'['+str(256+320+128+128)+r',384,192,384,48,128,128]'})
    nn.add_layer(['Mixed_5c'], {'name': 'AvgPool', 'type': 'AvgPool3D', 'kernel': r'[2, 7, 7]', 'stride': r'[1, 1, 1]'})
    nn.add_layer(['AvgPool'], {'name': 'Dropout', 'type': 'Dropout'})
    nn.add_layer(['Dropout'], {'name': 'Logits', 'type': 'Unit3D', 'channel': r'[%d, %d]'%(384+384+128+128, 101), 'kernel': r'[1, 1, 1]', 'stride': r'[1, 1, 1]', 'padding': r'0', 'normal, relu': 'False'})

    nn.show()
    nn.save('pdf')
