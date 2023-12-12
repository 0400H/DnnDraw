# -*- coding: UTF-8 -*-

# from abc import abstractmethod
import graphviz
import pickle
from copy import deepcopy


# https://github.com/xflr6/graphviz
class engine(object):
    def __init__(self):
        self.gv_gs = dict()
        self._edge_def = {
            'src' : '',
            'dst' : '',
            'label' : '',
            'attrs' : '',
        }
        self._node_def = {
            'name' : '',
            'label' : '',
            'attr' : {
                'shape' : 'record',
                'fontsize' : '8',
                'color' : 'lightblue2',
                'style' : 'filled',
            },
            'edges' : {},
        }
        self._graph_def = {
            'name' : '',
            'directed' : True,
            'attr' : {
                'engine' : 'dot',
                'rankdir' : 'BT',
                'size' : '100,100'
            },
            'node_attr' : self._node_def['attr'],
            'nodes' : {},
        }

    def get_def_attr(self, attr_type):
        path = '_' + attr_type +'_def'
        attr = getattr(self, path)
        return deepcopy(attr)

    def update_dict_attr(self, obj, attr):
        if type(attr) != dict:
            return attr
        else:
            for key in attr.keys():
                if key in obj.keys():
                    obj[key] = self.update_dict_attr(obj[key], attr[key])
                else:
                    obj[key] = attr[key]
            return obj

    def parse_def_obj(self, attr_type, attr):
        obj_def = self.get_def_attr(attr_type)
        return self.update_dict_attr(obj_def, attr)

    def add_graph(self, graph_attr):
        graph_name = graph_attr['name']
        graph_def = self.parse_def_obj('graph', graph_attr)
        if graph_def['directed'] == False:
            print('Add Graph {}'.format(graph_name))
            self.gv_gs[graph_name] = graphviz.Graph(graph_name)
        else:
            print('Add Digraph {}'.format(graph_name))
            self.gv_gs[graph_name] = graphviz.Digraph(graph_name)
        self.gv_gs[graph_name].attr(**graph_def['attr'])
        self.gv_gs[graph_name].node_attr.update(**graph_def['node_attr'])
        return graph_name

    def add_node(self, graph_name, node_attr, add_edges=True):
        node_name = node_attr['name']
        node_def = self.parse_def_obj('node', node_attr)
        self.gv_gs[graph_name].node(name=node_name, label=node_def['label'])
        print('Graph {} add Node {}'.format(graph_name, node_name))
        if add_edges:
            for src in node_def['edges']:
                self.add_edge(graph_name, {'src':src, 'dst':node_name})
        return node_name

    def add_edge(self, graph_name, edge_attr):
        edge_def = self.parse_def_obj('edge', edge_attr)
        self.gv_gs[graph_name].edge(tail_name=edge_def['src'],
                                  head_name=edge_def['dst'],
                                  label=edge_def['label'],
                                  attrs=edge_def['attrs'])
        print('Add Edge: {} -> {}'.format(edge_def['src'], edge_def['dst']))
        return True

    def merge_subgraph(self, root_graph_name, sub_graph_name):
        print('Graph {} add subgraph {}'.format(root_graph_name, sub_graph_name))
        self.gv_gs[root_graph_name].subgraph(self.gv_gs[sub_graph_name])
        return True

    def load(self, file_path):
        print('load Graph from file:', file_path)
        with open(file_path, 'rb')as fp:
            self.gv_gs = pickle.load(fp)

    def save(self, file_path):
        print('save Graph to file:', file_path)
        with open(file_path, 'wb')as fp:
            pickle.dump(self.gv_gs, fp)

    def gv_source(self, graph_name):
        return self.gv_gs[graph_name].source

    def gv_view(self, graph_name):
        self.gv_gs[graph_name].view()

    def gv_render(self, graph_name, format='svg'):
        print('save Graph {} to {} picture.'.format(graph_name, format))
        self.gv_gs[graph_name].format = format
        self.gv_gs[graph_name].render()


if __name__ == '__main__':
    graph_name = 'GraphViz_Test'
    graph_file = graph_name + '.pkl'
    proxy = engine()
    # import os
    # if os.path.exists(graph_file):
    #     proxy.load(graph_file)
    # else:
    if True:
        proxy.add_graph({'name':graph_name, 'directed':True})
        proxy.add_node(graph_name, {'name':'node1', 'label':'label1'})
        proxy.add_node(graph_name, {'name':'node2', 'label':'label2'})
        proxy.add_node(graph_name, {'name':'node3', 'label':'label3', 'edges': ['node1', 'node2']})
        proxy.add_graph({'name':'cluster_subgraph', 'directed':True})
        proxy.add_node('cluster_subgraph', {'name':'node4', 'label':'label4'})
        proxy.add_node('cluster_subgraph', {'name':'node5', 'label':'label5'})
        proxy.add_edge(graph_name, {'src':'node4', 'dst':'node5'})
        proxy.merge_subgraph(graph_name, 'cluster_subgraph')
        proxy.add_edge(graph_name, {'src':'node3', 'dst':'node4'})
        proxy.add_node(graph_name, {'name':'node6', 'label':'label6', 'edges': ['node3', 'node5']})
        proxy.save(graph_file)
    print(proxy.gv_source(graph_name))
    proxy.gv_view(graph_name)
    proxy.gv_render(graph_name, format='png')
