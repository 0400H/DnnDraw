## Examples


### Tinydnn

```python
import dnndraw

dnn = dnndraw.graph(name="tinydnn")

# first layer
dnn.add_node(in_nodes=[], node_info={'name': 'layer_1', 'Type': 'Conv3D', 'kernel': [1, 1, 1], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

dnn.add_node(in_nodes=['layer_1'], node_info={'name': 'layer_2', 'Type': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

dnn.add_node(in_nodes=['layer_1'], node_info={'name': 'layer_3', 'Type': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

# end layer
dnn.add_node(in_nodes=['layer_2', 'layer_3'], node_info={'name': 'layer_4', 'Type': 'Concat'})

dnn.save(format='png', file_path=dnn.name+'.json') # format: png, svg, pdf, ...
dnn.show()
```

![](https://raw.githubusercontent.com/AINoobs/repo_src/master/DnnDraw/tinydnn.gv.svg)

---

### [Facbook - DLRM](https://arxiv.org/abs/1906.00091)

![](https://raw.githubusercontent.com/AINoobs/repo_src/master/DnnDraw/DLRM.gv.svg)

---

### [DeepMind - I3D](https://arxiv.org/abs/1705.07750)

![](https://raw.githubusercontent.com/AINoobs/repo_src/master/DnnDraw/I3D_Topology.gv.svg)
