## DnnDraw

DnnDraw is a framework for visualizing neural networks via Python programming.

## How to install

Install from [pypi](https://pypi.org/project/dnndraw/):

```shell
pip install dnndraw
```

Install from source:

```shell
python -u ./setup.py bdist_wheel
pip install ./dist/*.whl
```

---

### [Examples](Examples.md)


#### Tinydnn

```python
import dnndraw

dnn = dnndraw.graph(name="tinydnn")

# first layer
dnn.add_node(in_nodes=[], node_info={'name': 'layer_1', 'Type': 'Conv3D', 'kernel': [1, 1, 1], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

dnn.add_node(in_nodes=['layer_1'], node_info={'name': 'layer_2', 'Type': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

dnn.add_node(in_nodes=['layer_1'], node_info={'name': 'layer_3', 'Type': 'Conv3D', 'kernel': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 'None', 'normal, relu': 'True'})

# end layer
dnn.add_node(in_nodes=['layer_2', 'layer_3'], node_info={'name': 'layer_4', 'Type': 'Concat'})

print(dnn.source())
dnn.export(format='png') # format: png, svg, pdf, ...
dnn.show()
```

![](https://raw.githubusercontent.com/AINoobs/repo_src/master/DnnDraw/tinydnn.gv.svg)

---

### Dev or Build Example

- via `develop` mode

    ```
    python3 ./setup.py develop
    cd example
    python ./tinydnn.py
    ```

- via `PYTHONPATH`

    ```
    export PYTHONPATH=`pwd`
    cd example
    python ./tinydnn.py
    ```