# coding:utf-8

from setuptools import setup, find_packages

setup(
    name = "dnndraw",
    version = "0.1.0",
    keywords = ["pip", "dnndraw", "dnn", "visualize"],
    description = "An editor that visualizes neural networks",
    long_description = "An editor that visualizes neural networks",
    license = "MIT Licence",
    url = "https://github.com/0400H/DnnDraw",
    author = "0400h",
    author_email = "git@0400h.cn",
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["graphviz", "pillow"]
)
