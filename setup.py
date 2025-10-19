# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup

import fastentrypoints

dependencies = [
    "click",
    "matplotlib",
    "vispy",
    "polyscope",
    "pint",
    "clicktool @ git+https://git@github.com/jakeogh/clicktool",
]
# "datoviz",

config = {
    "version": "0.1",
    "name": "mptuple3d",
    "url": "https://github.com/jakeogh/mptuple3d",
    "license": "ISC",
    "author": "Justin Keogh",
    "author_email": "github.com@v6y.net",
    "description": "plots 3-tuples in 3D space",
    "long_description": __doc__,
    "packages": find_packages(exclude=["tests"]),
    "package_data": {"mptuple3d": ["py.typed"]},
    "include_package_data": True,
    "zip_safe": False,
    "platforms": "any",
    "install_requires": dependencies,
    "entry_points": {
        "console_scripts": [
            "mptuple3d=mptuple3d.cli:cli",
        ],
    },
}

setup(**config)
