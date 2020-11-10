import ast
import pathlib
from setuptools import setup

def readme():
    with open("README.rst") as f:
        return f.read()

def get_version():
    with open(r"oval/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[-1].strip()
    return ast.literal_eval(version)

setup(
    name="Oval",
    version=get_version(),
    description="Understand the world from option trader's perspective",
    long_description=readme(),
    keywords="Options Valuation",
    url="",
    author="Kenny Li",
    author_email="Kenseng1596357@gmail.com",
    license="BSD3",
    package=["oval"],
    entry_points={},
    include_package_data=True,
    zip_safe=False,
)  