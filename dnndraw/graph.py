# -*- coding: UTF-8 -*-

from .engine import engine


class graph(object):
    def __init__(self, name):
        self.name = name
        self.engine = engine()
        self.add_graph(self.name, True)
        pass

    def format_info(self, node_info):
        if type(node_info) == dict:
            key_info = "{{{}}}".format("|".join(node_info.keys()))
            value_list = []
            for key in node_info:
                value_list.append(self.format_info(node_info[key]))
            value_info = "{{{}}}".format("|".join(value_list))
            fmt_str = key_info + "|" + value_info
            return fmt_str
        elif type(node_info) == tuple:
            value_list = []
            for value in node_info:
                value_list.append(self.format_info(value))
            fmt_str = r"({})".format(", ".join(value_list))
            return fmt_str
        elif type(node_info) == list:
            value_list = []
            for value in node_info:
                value_list.append(self.format_info(value))
            fmt_str = r"[{}]".format(", ".join(value_list))
            return fmt_str
        else:
            return str(node_info)

    def add_graph(self, graph_name, directed=True, subgraph=False):
        if subgraph:
            graph_name = 'cluster_' + graph_name
        graph_def = {
            'name' : graph_name,
            'directed' : directed,
            'label': '',
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
            'label' : self.format_info(node_info),
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

    def source(self):
        return self.engine.gv_source(self.name)

    def export(self, format='svg'):
        self.engine.gv_render(self.name, format)

    def show(self):
        self.engine.gv_view(self.name)

    def dump(self, file_path=None):
        if not file_path:
            file_path = self.name + '.pkl'
        self.engine.dump(file_path)

    def load(self, file_name):
        self.engine.load(file_name)
