# -*- coding: UTF-8 -*-

from .engine import engine


class graph(object):
    def __init__(self, name, layout='TB', node_attr="def"):
        self.name = name
        self.engine = engine(layout)
        self.add_graph(self.name, directed=True, subgraph=False, graph_attr={'rankdir': layout}, node_attr='def')
        self.kv_pair = False if layout == 'TB' or layout == 'BT' else True
        pass

    def fmt_to_polygon_label(self, node_info):
        """Format node information into a label string compatible with Mermaid"""
        if type(node_info) == dict:
            label_parts = []
            label_str = "<\n<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">\n{}\n</TABLE>\n>"
            for k, v in node_info.items():
                if k not in ['name', 'edges']:  # Skip specific keys that aren't part of the label
                    formatted_value = self.fmt_to_polygon_label(v)
                    label_parts.append(f"<TR><TD ALIGN=\"LEFT\">{k}: {formatted_value}</TD></TR>")
            return label_str.format("\n".join(label_parts))
        elif type(node_info) == tuple or type(node_info) == list:
            value_list = []
            str_fmt = "[{}]"
            if type(node_info) == tuple:
                str_fmt = "({})"
            for value in node_info:
                value_list.append(self.fmt_to_polygon_label(value))
            fmt_str = str_fmt.format(", ".join(value_list))
            return fmt_str
        else:
            return str(node_info)

    def add_graph(self, graph_name, directed=True, subgraph=False, graph_attr={}, node_attr={}):
        if subgraph:
            graph_name = 'cluster_' + graph_name
        graph_def = {
            'name': graph_name,
            'directed': directed,
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

    def add_node(self, in_nodes, node_info, graph_name=None):
        '''
        in_nodes: ['node_name_1', 'node_name_2', ...] or {src: label, ...}
        node_info: {'name':'node_name', 'label':'node_info'}
        '''
        if graph_name == None:
            graph_name = self.name

        # Format the label for Mermaid
        formatted_label = node_info.get('label', '')
        if not formatted_label:
            formatted_label = self.fmt_to_polygon_label(node_info)

        node_def = {
            'name': node_info['name'],
            'label': formatted_label,
        }

        self.engine.add_node(graph_name, node_def)

        # Add edges from in_nodes to this node
        if in_nodes:
            if isinstance(in_nodes, list):
                for src in in_nodes:
                    edge_attr = {'src': src, 'dst': node_info['name']}
                    self.engine.add_edge(self.name, edge_attr)
            elif isinstance(in_nodes, dict):
                for src, label in in_nodes.items():
                    edge_attr = {'src': src, 'dst': node_info['name'], 'label': label}
                    self.engine.add_edge(self.name, edge_attr)
        return node_info['name']

    def merge_subgraph(self, root_graph_name, sub_graph_name):
        if 'cluster_' not in sub_graph_name:
            sub_graph_name = 'cluster_' + sub_graph_name
        return self.engine.merge_subgraph(root_graph_name, sub_graph_name)

    def source(self):
        return self.engine.source(self.name)

    def export(self, format='svg'):
        self.engine.render(self.name, format)

    def show(self):
        self.engine.view(self.name)

    def dump(self, file_path=None):
        if not file_path:
            file_path = self.name + '.pkl'
        self.engine.dump(file_path)

    def load(self, file_name):
        self.engine.load(file_name)
