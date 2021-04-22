# -*- coding: UTF-8 -*-

from .engine import engine

class graph(object):
    def __init__(self, name, size, out_format='svg'):
        self.engine = engine()
        self.graph_name = name
        self.engine.add_graph(self.graph_name, True, size, 'lightblue2', 'filled', out_format)
        pass

    def format_info_dict(self, info_dict):
        key_info = r'{'
        value_info = r'{'
        for key in info_dict:
            key_info = key_info + key + '|'
            value_info = value_info + str(info_dict[key]) + '|'
        info = key_info[:-1] + r'} | ' + value_info[:-1] + r'}'
        return info

    def add_layer(self, in_layers, info_dict):
        layer = {}
        layer['name'] = info_dict['name']
        layer['label'] = self.format_info_dict(info_dict)
        layer['shape'] = 'record'
        return self.engine.graph_add_node(self.graph_name, in_layers, layer)

    def show(self):
        return self.engine.graph_view(self.graph_name)

    def save(self, out_format):
        return self.engine.graph_render(self.graph_name, out_format)
pass