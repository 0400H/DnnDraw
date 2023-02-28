# -*- coding: UTF-8 -*-

from .engine import engine


class graph(object):
    def __init__(self, name):
        self.name = name
        self.engine = engine()
        self.add_graph(self.name, True)
        pass

    def record_format(self, info_dict):
        key_info = r'{'
        value_info = r'{'
        for key in info_dict:
            key_info = key_info + key + '|'
            value_info = value_info + str(info_dict[key]) + '|'
        info = key_info[:-1] + r'} | ' + value_info[:-1] + r'}'
        return info

    def add_graph(self, graph_name, directed=False):
        graph_def = {
            'name' : graph_name,
            'directed' : directed,
        }
        return self.engine.add_graph(graph_def)

    # https://www.graphviz.org/doc/info/shapes.html
    def add_node(self, in_nodes, node_info, color='lightblue2', style='filled'):
        '''
        in_nodes: ['node_name_1', 'node_name_2', ...]
        node_attr: {'name':'node_name', 'label':'node_info'}
        '''
        node_attr = {
            'name' : node_info['name'],
            'attr' : {
                'shape' : 'record',
                'fontsize' : '8',
                'color' : color,
                'style' : style,
            },
            'label' : self.record_format(node_info),
            'edges' : [name for name in in_nodes]
        }
        return self.engine.add_node(self.name, node_attr)

    def print(self):
        print(self.engine.graph_dumps(self.name))

    def show(self):
        self.engine.gv_view(self.name)

    def load(self, file_name):
        self.engine.graph_load(file_name)

    def save(self, format='svg', file_path=None):
        if not file_path:
            file_path = self.name + '.json'
        self.engine.graph_save(self.name, file_path)
        self.engine.gv_render(self.name, format)
pass