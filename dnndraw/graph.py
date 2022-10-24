# -*- coding: UTF-8 -*-

from .engine import engine

class graph(object):
    def __init__(self, name, size='100,100', out_format='svg'):
        self.name = name
        self.engine = engine()
        self.engine.add_graph(self.name, True, size, 'lightblue2', 'filled', out_format)
        pass

    def format(self, info_dict):
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
        layer['label'] = self.format(info_dict)
        layer['shape'] = 'record'
        return self.engine.graph_add_node(self.name, in_layers, layer)

    def show(self):
        return self.engine.graph_view(self.name)

    def save(self, out_format):
        return self.engine.graph_render(self.name, out_format)
pass