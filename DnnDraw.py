# -*- coding: UTF-8 -*-

from GraphViz import GraphViz

class DnnDraw (GraphViz):
    def __init__(self, name, size, out_format='svg'):
        GraphViz.__init__(self)
        self._dnn_name = name
        self.info_list = []
        self.add_graph(self._dnn_name, True, size, 'lightblue2', 'filled', out_format)
        return None

    def info_format(self, info_dict):
        self.info_list.append(info_dict)
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