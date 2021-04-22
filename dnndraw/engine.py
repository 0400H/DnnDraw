# -*- coding: UTF-8 -*-

# from abc import abstractmethod
import graphviz

# https://graphviz.org/download
class engine(object):
    def __init__(self):
        self.graphs = dict()
        return None

    def add_graph(self, graph_name, directed=False, graph_size='8,5', node_color='lightblue2', graph_style='filled', out_format='pdf'):
        if graph_name in self.graphs:
            print('graph {} is already exists!'.format(graph_name))
            return False
        else:
            if directed == False:
                self.graphs[graph_name] = graphviz.Graph(graph_name, format=out_format)
            else:
                self.graphs[graph_name] = graphviz.Digraph(graph_name, format=out_format)

            self.graphs[graph_name].node_attr.update(color=node_color, shape= 'record', style=graph_style)
            self.graphs[graph_name].attr(size=graph_size)
            return True
        pass

    # https://www.graphviz.org/doc/info/shapes.html
    def graph_add_node(self, graph_name, in_nodes, node):
        '''
        in_nodes: ['node_name_1', 'node_name_2', ...]
        node: {'name':'node_name', 'label':'label_info', 'shape':'box', 'fontsize'='8'}
        shape: 'circle', 'box', 'record'
        '''
        if node['name'] in self.graphs.keys():
            print('node {} is already exists in graph {}!'.format(node['name'], graph_name))
            return False

        if in_nodes != None and type(in_nodes) != type([]):
            print('wrong in_nodes type!')
            return False

        fontsize = '8' if 'fontsize' not in node.keys() else node['fontsize']
        self.graphs[graph_name].node(name=node['name'], \
                                     label=node['label'], \
                                     shape=node['shape'], \
                                     fontsize=fontsize)

        if in_nodes != None:
            for node_name in in_nodes:
                self.graphs[graph_name].edge(node_name, node['name'])
        return True
    pass

    def graph_view(self, graph_name):
        self.graphs[graph_name].view()
        return None

    def graph_render(self, graph_name, out_format='svg'):
        self.graphs[graph_name].format = out_format
        self.graphs[graph_name].render()
        return None

    def save_resource(self, graph_name, file_path):
        self.graphs[graph_name].write(file_path)
        return None

    def save_picture(self, graph_name, file_path):
        self.graphs[graph_name].draw(file_path)
        return None

if __name__ == '__main__':
    graph_name = 'GraphViz Test'
    proxy = engine()
    proxy.add_graph(graph_name, True, '100,100', 'lightblue2', 'filled', 'svg')
    proxy.graph_add_node(graph_name, None, {'name':'name1', 'label':'label1', 'shape':'box'}, )
    proxy.graph_add_node(graph_name, None, {'name':'name2', 'label':'label2', 'shape':'box'}, )
    proxy.graph_add_node(graph_name, ['name1', 'name2'], {'name':'name3', 'label':'label3', 'shape':'box'})
    proxy.graph_view(graph_name)
    proxy.graph_render(graph_name)