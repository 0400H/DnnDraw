# -*- coding: UTF-8 -*-

# from abc import abstractmethod
import graphviz
import json
from copy import deepcopy


# https://github.com/xflr6/graphviz
class engine(object):
    def __init__(self):
        self.gvs = dict()
        self.graphs = dict()
        self._edge_def = {
            'name' : '',                 # in node name
            'label' : '',
            'attrs' : '',
        }
        self._node_def = {
            'name' : '',
            'attr' : {
                'shape' : 'box',
                'fontsize' : '8',
                'color' : 'lightblue2',
                'style' : 'filled',
            },
            'label' : '',
            'edges' : {},
        }
        self._graph_def = {
            'name' : '',
            'directed' : True,
            'attr' : {
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
        if graph_name in self.graphs.keys():
            print('graph {} is already exists!'.format(graph_name))
            return False

        # graph level
        graph_def = self.parse_def_obj('graph', graph_attr)
        self.graphs[graph_name] = deepcopy(graph_def)
        self.graphs[graph_name]['nodes'] = {}
        if graph_def['directed'] == False:
            print('add Graph: {}'.format(graph_name))
            self.gvs[graph_name] = graphviz.Graph(graph_name)
        else:
            print('add Digraph: {}'.format(graph_name))
            self.gvs[graph_name] = graphviz.Digraph(graph_name)
        self.gvs[graph_name].attr(**graph_def['attr'])
        self.gvs[graph_name].node_attr.update(**graph_def['node_attr'])

        # node level
        nodes = graph_def['nodes']
        for name in nodes:
            self.add_node(graph_name, nodes[name])
        return True

    def add_node(self, graph_name, node_attr):
        try:
            self.graphs[graph_name]
        except Exception as e:
            print('Graph {} is not exists, please add it first!'.format(graph_name))
            return False

        node_name = node_attr['name']
        if node_name in self.graphs[graph_name]['nodes'].keys():
            print('Node {} is already exists in Graph {}!'.format(node_name, graph_name))
            return False

        # node level
        print('add Node: {}'.format(node_name))
        node_def = self.parse_def_obj('node', node_attr)
        self.graphs[graph_name]['nodes'][node_name] = deepcopy(node_def)
        self.graphs[graph_name]['nodes'][node_name]['edges'] = {}
        self.gvs[graph_name].node(name=node_name, **node_def['attr'])

        # edge level
        edges = node_def['edges']
        edges_attr = {}
        if type(edges) == list:
            for name in edges:
                edges_attr[name] = {'name' : name}
        else:
            edges_attr = edges
        for name in edges:
            edges_attr[name]['name'] = name
            self.add_edge(graph_name, node_name, edges_attr[name])
        return True

    def add_edge(self, graph_name, node_name, edge_attr):
        try:
            self.graphs[graph_name]['nodes'][node_name]
        except Exception as e:
            print('Node {} is not exists in Graph {}, please add it first!'.format(node_name, graph_name))
            return False

        edge_name = edge_attr['name']
        if edge_name in self.graphs[graph_name]['nodes'][node_name]['edges'].keys():
            print('Edge {} is already exists in Node {}/{}!'.format(edge_name, graph_name, node_name))
            return False

        # edge level
        print('add Edge: {} -> {}'.format(edge_name, node_name))
        edge_def = self.parse_def_obj('edge', edge_attr)
        self.graphs[graph_name]['nodes'][node_name]['edges'][edge_def['name']] = deepcopy(edge_def)
        self.gvs[graph_name].edge(tail_name=edge_name,
                                  head_name=node_name,
                                  label=edge_def['label'],
                                  attrs=edge_def['attrs'])
        return True

    def graph(self, graph_name):
        return self.graphs[graph_name]

    def graph_dumps(self, graph_name):
        return json.dumps(self.graph(graph_name), indent=1)

    def graphs_dumps(self):
        return json.dumps(self.graphs, indent=1)

    def graph_load(self, file_path):
        with open(file_path, 'r') as fp:
            print('load Graph from file:', file_path)
            graph = json.loads(fp.read())
            self.add_graph(graph)

    def graphs_load(self, file_path):
        with open(file_path, 'r') as fp:
            print('load Graphs from file:', file_path)
            graphs = json.loads(fp.read())
            for name in graphs:
                self.add_graph(graphs[name])

    def graph_save(self, graph_name, file_path):
        with open(file_path, 'w') as fp:
            print('save Graph to file:', file_path)
            fp.write(self.graph_dumps(graph_name))

    def graphs_save(self, file_path):
        with open(file_path, 'w') as fp:
            print('save Graphs to file:', file_path)
            fp.write(self.graphs_dumps())

    def gv_source(self, graph_name):
        return self.gvs[graph_name].source

    def gv_view(self, graph_name):
        self.gvs[graph_name].view()

    def gv_render(self, graph_name, format='svg'):
        print('save Graph {} to {} picture.'.format(graph_name, format))
        self.gvs[graph_name].format = format
        self.gvs[graph_name].render()


if __name__ == '__main__':
    graph_name = 'GraphViz Test'
    graph_json = graph_name+'.json'
    proxy = engine()
    import os
    if os.path.exists(graph_json):
        proxy.graph_load(graph_json)
    else:
    # if True:
        proxy.add_graph({'name':graph_name, 'directed':True})
        proxy.add_node(graph_name, {'name':'node1', 'label':'label1'})
        proxy.add_node(graph_name, {'name':'node2', 'label':'label2'})
        proxy.add_node(graph_name, {'name':'node3', 'label':'label3', 'edges': ['node1', 'node2']})
        # proxy.add_node(graph_name, {'name':'node3', 'label':'label3', 'edges': {'node1':{}, 'node2':{}}})
        proxy.graph_save(graph_name, graph_name+'.json')
    print(proxy.graphs_dumps())
    print(proxy.gv_source(graph_name))
    proxy.gv_view(graph_name)
    proxy.gv_render(graph_name, format='png')
