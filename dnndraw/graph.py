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

    def add_graph(self, graph_name, directed=True, subgraph=False):
        if subgraph:
            graph_name = 'cluster_' + graph_name
        graph_def = {
            'name' : graph_name,
            'directed' : directed,
        }
        self.engine.add_graph(graph_def)
        return graph_name

    # https://www.graphviz.org/doc/info/shapes.html
    def add_node(self, in_nodes, node_info, color='lightblue2', style='filled', graph_name=None):
        '''
        in_nodes: ['node_name_1', 'node_name_2', ...]
        node_attr: {'name':'node_name', 'label':'node_info'}
        '''
        if graph_name == None:
            graph_name = self.name
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
        self.engine.add_node(graph_name, node_attr, add_edges=False)
        for src in node_attr['edges']:
            self.engine.add_edge(self.name, {'src':src, 'dst':node_info['name']})
        return True

    def merge_subgraph(self, root_graph_name, sub_graph_name):
        if 'cluster_' not in sub_graph_name:
            sub_graph_name = 'cluster_' + sub_graph_name
        return self.engine.merge_subgraph(root_graph_name, sub_graph_name)

    def show(self, format='svg'):
        self.engine.gv_render(self.name, format)
        self.engine.gv_view(self.name)

    def load(self, file_name):
        self.engine.load(file_name)

    def save(self, file_path=None):
        if not file_path:
            file_path = self.name + '.pkl'
        self.engine.save(file_path)
