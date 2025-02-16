# -*- coding: UTF-8 -*-

from .engine import engine


class graph(object):
    def __init__(self, name, layout='TB', node_attr="def"):
        self.name = name
        self.engine = engine(layout)
        self.add_graph(self.name, directed=True, subgraph=False, graph_attr={'rankdir':layout}, node_attr='def')
        self.kv_pair = False if layout == 'TB' or layout == 'BT' else True
        pass

    def fmt_to_polygon_label(self, node_info):
        if type(node_info) == dict:
            k_max_len, v_max_len = 0, 0
            kv_list = []
            v_list = []
            for k,v in node_info.items():
                v_list.append(self.fmt_to_polygon_label(v))
                k_max_len = max(k_max_len, len(k))
                v_max_len = max(v_max_len, len(v_list[-1]))
            str_fmt = f"{{:<{k_max_len}}} {{:<{v_max_len}}}"
            for i,k in enumerate(node_info):
                fmt_str = str_fmt.format(k, v_list[i])
                kv_list.append(fmt_str)
            fmt_str = "\l".join(kv_list) + "\l"
            return fmt_str
        elif type(node_info) == tuple or type(node_info) == list:
            value_list = []
            str_fmt = r"[{}]"
            if type(node_info) == tuple:
                str_fmt = r"({})"
            for value in node_info:
                value_list.append(self.fmt_to_polygon_label(value))
            fmt_str = str_fmt.format(", ".join(value_list))
            return fmt_str
        else:
            return str(node_info)

    def fmt_to_record_label(self, node_info, kv_pair=False):
        if type(node_info) == dict:
            if kv_pair:
                pairs = []
                for k, v in node_info.items():
                    pairs.append("{{{}|{}}}".format(k, v))
                fmt_str = "|".join(pairs)
            else:
                key_info = "{{{}}}".format("|".join(node_info.keys()))
                value_list = []
                for key in node_info:
                    value_list.append(self.fmt_to_record_label(node_info[key]))
                value_info = "{{{}}}".format("|".join(value_list))
                fmt_str = key_info + "|" + value_info
            return fmt_str
        else:
            return self.fmt_to_polygon_label(node_info)

    def add_graph(self, graph_name, directed=True, subgraph=False, graph_attr={}, node_attr={}):
        if subgraph:
            graph_name = 'cluster_' + graph_name
        graph_def = {
            'name' : graph_name,
            'directed' : directed,
        }
        if graph_attr != {}:
            if graph_attr == "def":
                graph_def["attr"] = self.engine.get_def("graph")["attr"]
            else:
                graph_def["attr"] = graph_attr
        if node_attr != {}:
            if node_attr == "def":
                graph_def["node_attr"] = self.engine.get_def("graph")["node_attr"]
            else:
                graph_def["node_attr"] = node_attr
        self.engine.add_graph(graph_def)
        return graph_name

    # https://www.graphviz.org/doc/info/shapes.html
    def add_node(self, in_nodes, node_info, graph_name=None):
        '''
        in_nodes: ['node_name_1', 'node_name_2', ...]
        node_info: {'name':'node_name', 'label':'node_info'}
        '''
        if graph_name == None:
            graph_name = self.name
        node_def = {
            'name' : node_info['name'],
            'label' : self.fmt_to_polygon_label(node_info),
        }
        self.engine.add_node(graph_name, node_def)
        if type(in_nodes) == list:
            for src in in_nodes:
                edge_attr = {'src':src, 'dst':node_info['name']}
                self.engine.add_edge(self.name, edge_attr)
        else:
            for src in in_nodes:
                edge_attr = {'src':src, 'dst':node_info['name'], 'label':in_nodes[src]}
                self.engine.add_edge(self.name, edge_attr)
        return node_info['name']

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
