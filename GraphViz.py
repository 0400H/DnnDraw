# -*- coding: UTF-8 -*-

import graphviz as gv

class GraphViz (object):
    def __init__(self):
        self._node = set()
        self.graph = dict()
        return None

    def view(self, graph_name):
        self.graph[graph_name].view()
        return None

    def render(self, graph_name, out_format='svg'):
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
            self.graph[graph_name] = gv.Graph(graph_name, format=out_format)
        else:
            self.graph[graph_name] = gv.Digraph(graph_name, format=out_format)
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

        if node_in != None and type(node_in) != type([]):
            print('wrong node_in type!')
            return False

        if node_in != None:
            for in_layer in node_in:
                self.graph[graph_name].edge(in_layer, name)

        return True

pass

if __name__ == '__main__':
    graph_name = 'GraphViz Test'
    graph = GraphViz()
    graph.add_graph(graph_name, True, '100,100', 'lightblue2', 'filled', 'svg')
    graph.add_node(graph_name, None, 'name1', 'label1', 'box')
    graph.add_node(graph_name, ['name1'], 'name2', 'label2', 'box')
    graph.view(graph_name)
    graph.render(graph_name)