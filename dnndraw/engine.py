# -*- coding: UTF-8 -*-

# from abc import abstractmethod
import graphviz as gv
from copy import deepcopy


# https://github.com/xflr6/graphviz
class graphviz_engine(object):
    def __init__(self):
        self.graph = None
        self.name = None
        self.graph_attr = {
            'engine': 'dot',     # neato, fdp
            # 'style': 'invis',
            'rankdir': 'TB',     # only for major graph, mode: TB, BT, LR, RL https://graphviz.org/docs/attrs/rankdir/
            # 'rank': 'same',     # only for sub graph, https://graphviz.org/docs/attrs/rank/
            # 'dpi': '100',
        }
        self.node_attr = {
            'shape': 'box', # https://graphviz.org/doc/info/shapes.html
            'style': 'filled',
            'color': '#F2E9FF',
            'fillcolor': '#F2E9FF',
            'penwidth': '1',
            # 'fontname': 'Courier'
            # 'fontsize': '8',
        }
        self.edge_attr = ''

    def update_attr_dict(self, raw_attr, new_attr):
        attr = deepcopy(raw_attr)
        attr.update(new_attr)
        return attr

    def create_graph(self, name, directed=True, attr=None):
        print('Create Graph {}.'.format(name))
        if directed:
            g = gv.Digraph(name=name)
        else:
            g = gv.Graph(name=name)
        if attr:
            g.attr(**attr)
        return g

    def create_root_graph(self, name, directed=True, attr=None):
        self.name = name
        self.graph = self.create_graph(name, directed, attr)
        return self.graph

    def send_to_merge_stack(self, event):
        if len(event) == 2:
            self.merge_stack[0].append(event)
        else:
            self.merge_stack[1].append(event)

    def create_sub_graph(self, name, directed=True, has_border=False, attr=None):
        if has_border:
            cg = self.create_graph('cluster_'+name, directed)
            g = self.create_graph(name, directed, attr)
            return cg, g
        else:
            g = self.create_graph(name, directed, attr)
            return g

    def merge_graph(self, root_graph, sub_graph):
        root_graph.subgraph(sub_graph)
        print('Graph {} merge graph {}.'.format(root_graph.name, sub_graph.name))
        return root_graph

    def graph_add_node(self, graph, name, label=None, attr={}):
        print('Graph {} add node {}.'.format(graph.name, name))
        if label:
            graph.node(name=name, label=label, **attr)
        else:
            graph.node(name=name, **attr)
        return name

    def add_node(self, name, label=None, attr={}):
        self.graph_add_node(self.graph, name=name, label=label, attr=attr)
        return name

    def graph_add_edge(self, graph, src, dst, label='', attr=''):
        print('Graph {} add edge {} -> {}.'.format(graph.name, src, dst))
        if label:
            if attr:
                graph.edge(tail_name=src, head_name=dst, label=label, attrs=attr)
            else:
                graph.edge(tail_name=src, head_name=dst, label=label)
        else:
            if attr:
                graph.edge(tail_name=src, head_name=dst, attrs=attr)
            else:
                graph.edge(tail_name=src, head_name=dst)

    def add_edge(self, src, dst, label='', attr=''):
        self.graph_add_edge(self.graph, src=src, dst=dst, label=label, attr=attr)

    def source(self, graph=None):
        if graph is None:
            graph = self.graph
        return graph.source

    def from_source(self, file_path):
        print(f'load Graph from {file_path}.')
        with open(file_path, 'r') as fp:
            gv_src = fp.read()
            g = gv.Source(gv_src)
            return g

    def render(self, format='svg', graph=None):
        if graph is None:
            graph = self.graph
        print('export Graph {} to {} format.'.format(graph.name, format))
        graph.format = format
        graph.render(view=False)

    def view(self, graph=None):
        if graph is None:
            graph = self.graph
        graph.view()

if __name__ == '__main__':
    e = graphviz_engine()
    g = e.create_root_graph('SubGraph', directed=True, attr=e.graph_attr)

    node_color = [
        {"fillcolor":"#E5F6FF", "color": "#73A6FF"},
        {"fillcolor":"#FFF6CC", "color": "#FFBC52"},
        {"fillcolor":"#FFEBEB", "color": "#E68994"},
        {"fillcolor":"#E5F6FF", "color": "#73A6FF"},
        {"fillcolor":"#D5F5E3", "color": "#73C6B6"},
        {"fillcolor":"#F2E9FF", "color": "#B39DDB"},
    ]

    # subgraph
    cg1, g1 = e.create_sub_graph('g1', directed=True, has_border=True)
    A = e.graph_add_node(g1, 'A', attr=e.update_attr_dict(e.node_attr, node_color[0]))
    B = e.graph_add_node(g1, 'B', attr=e.update_attr_dict(e.node_attr, node_color[1]))
    e.graph_add_edge(g1, A, B)
    e.merge_graph(cg1, g1)
    e.merge_graph(g, cg1)

    # https://graphviz.readthedocs.io/en/stable/examples.html#rank-same-py
    cg2, g2 = e.create_sub_graph('g2', directed=True, has_border=True, attr={'rank':'same'})
    C = e.graph_add_node(g2, 'C', attr=e.update_attr_dict(e.node_attr, node_color[2]))
    D = e.graph_add_node(g2, 'D', attr=e.update_attr_dict(e.node_attr, node_color[3]))
    e.graph_add_edge(g2, C, D)
    e.merge_graph(cg2, g2)
    e.merge_graph(g, cg2)

    cg3, g3 = e.create_sub_graph('g3', directed=True, has_border=True)
    E = e.graph_add_node(g3, 'E', attr=e.update_attr_dict(e.node_attr, node_color[4]))
    F = e.graph_add_node(g3, 'F', attr=e.update_attr_dict(e.node_attr, node_color[5]))
    e.graph_add_edge(g3, E, F)
    e.merge_graph(cg3, g3)
    e.merge_graph(g, cg3)

    e.add_edge(A, C)
    e.add_edge(B, E)
    e.add_edge(D, E)


    print(e.source())
    print(e.render("svg"))
    # e.graph
