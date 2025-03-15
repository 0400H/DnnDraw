# -*- coding: UTF-8 -*-

from .engine import graphviz_engine
import random

class graph(object):
    def __init__(self, name, layout='TB', node_attr={}, label_align=True):
        self.name = name
        self.engine = graphviz_engine()
        self.engine.create_root_graph(
            name, directed=True,
            attr=self.engine.update_attr_dict(
                    self.engine.graph_attr, {'rankdir': layout})
        )
        self.node_attr = node_attr
        self.node_colors = [
            {"fillcolor":"#E5F6FF", "color": "#73A6FF"},
            {"fillcolor":"#FFF6CC", "color": "#FFBC52"},
            {"fillcolor":"#FFEBEB", "color": "#E68994"},
            {"fillcolor":"#D5F5E3", "color": "#73C6B6"},
            {"fillcolor":"#F2E9FF", "color": "#B39DDB"},
        ]
        self.label_align = label_align

    def fmt_to_polygon_label(self, node_info):
        if type(node_info) == dict:
            label_parts = []
            if self.label_align:
                label_str = "<\n<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">\n{}\n</TABLE>\n>"
            else:
                label_str = "{}"
            for k, v in node_info.items():
                formatted_value = self.fmt_to_polygon_label(v)
                if self.label_align:
                    label_parts.append(f"<TR><TD ALIGN=\"LEFT\">{k}: {formatted_value}</TD></TR>")
                else:
                    label_parts.append(f"{k}: {formatted_value}")
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

    def create_graph(self, name, directed=True, has_border=False, attr=None):
        return self.engine.create_sub_graph(name, directed, has_border, attr)

    def merge_graph(self, sub_graph, root_graph=None):
        if not root_graph:
            root_graph = self.engine.graph
        return self.engine.merge_graph(root_graph, sub_graph)

    def add_node(self, in_nodes, node_info, node_attr=None, graph=None, rand_color=False):
        '''
        in_nodes: ['node_name_1', 'node_name_2', ...] or {in_node1: label1, ...}
        node_info: {'name':'node_name', 'label':'node_info', 'attr': attr}
        '''
        if graph == None:
            graph = self.engine.graph

        name = node_info['name']
        label = self.fmt_to_polygon_label(node_info)
        node_attr = node_attr if node_attr else self.node_attr
        attr = self.engine.update_attr_dict(self.engine.node_attr, node_attr)
        if rand_color:
            attr = self.engine.update_attr_dict(self.engine.node_attr, random.choice(self.node_colors))
        self.engine.graph_add_node(graph, name, label, attr)

        # Add edges from in_nodes to this node
        if in_nodes:
            if isinstance(in_nodes, list):
                for src in in_nodes:
                    self.engine.add_edge(src, name)
            elif isinstance(in_nodes, dict):
                for src, label in in_nodes.items():
                    self.engine.add_edge(src, name, label)
        return name

    def source(self):
        return self.engine.source()

    def export(self, format='svg'):
        self.engine.render(format)

    def show(self):
        self.engine.view()
