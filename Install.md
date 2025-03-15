# Development

## How to install

```
apt-get install graphviz graphviz-dev
```

Install from source:

```shell
pip install git+https://github.com/0400H/DnnDraw.git
```

Install from [pypi](https://pypi.org/project/dnndraw/):

```shell
pip install dnndraw
```

## How to develop and test example

- via `PYTHONPATH`

    ```
    export PYTHONPATH=`pwd`
    cd example
    python ./tinydnn.py
    ```

- via `editable` mode

    ```
    pip install -e .
    cd example
    python ./tinydnn.py
    ```

- via `develop` mode

    ```
    python3 ./setup.py develop
    cd example
    python ./tinydnn.py
    ```
