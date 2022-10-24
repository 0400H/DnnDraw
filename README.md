### DnnDraw

DnnDraw is an editor that visualizes neural networks.

Example - tinydnn

- Code

    ```python
    import dnndraw

    dnn = dnndraw.graph(name="tinydnn", out_format='svg')

    # first layer
    dnn.add_layer(in_layers=[], info_dict={'name': 'layer_1', 'Type': 'Conv3D', 'kernel': [1, 1, 1], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

    dnn.add_layer(in_layers=['layer_1'], info_dict={'name': 'layer_2', 'Type': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

    dnn.add_layer(in_layers=['layer_1'], info_dict={'name': 'layer_3', 'Type': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

    # end layer
    dnn.add_layer(in_layers=['layer_2', 'layer_3'], info_dict={'name': 'layer_4', 'Type': 'Concat'})

    dnn.show()
    dnn.save('png')
    ```

- Graph

    ![tinydnn](
    https://raw.githubusercontent.com/AINoobs/repo_src/master/DnnDraw/tinydnn.gv.svg)

Example - [I3D](https://arxiv.org/abs/1705.07750)

- Graph

    ![I3D](https://raw.githubusercontent.com/AINoobs/repo_src/master/DnnDraw/I3D_Topology.gv.svg)
