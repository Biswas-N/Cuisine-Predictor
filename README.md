# Norman PD Incidents Parser (Project 0)
## Developer: Biswas Nandamuri
Norman PD Incidents Parser is a python based utillity tool used to extract incidents data from a provided incident PDF file URL.

> The project's python code follows PEP8 Style Guide

This utility uses a number of open source projects:

* [PyPDF2](https://github.com/mstamy2/PyPDF2) - Utility to read and write PDFs with Python
* [Pytest](https://github.com/pytest-dev/pytest) - Testing framework that supports complex functional testing
* [Pytest-cov](https://github.com/pytest-dev/pytest-cov) - Coverage plugin for pytest
* [Pandas](https://github.com/pandas-dev/pandas) - Flexible and powerful data analysis / manipulation library for Python
* [Jupyterlab](https://github.com/jupyterlab/jupyterlab) - Browser-based computational environment for python
* [autopep8](https://github.com/hhatto/autopep8) - Tool that automatically formats Python code to conform to the PEP 8 style guide

## Run on local system
1. Clone this repository and move into the folder.
    ```sh
    $ git clone https://github.com/Biswas-N/cs5293sp22-project0.git
    $ cd cs5293sp22-project0
    ```
2. Install dependencies using [Pipenv](https://github.com/pypa/pipenv).
    ```sh
    $ pipenv install
    ``` 
3. Run the utility tool
    ```sh
    $ make
    ```
   > Note: Project includes a `Makefile` which has commonly used commands. By running `make` the following command `pipenv run python main.py --incidents <Sample URL>'` is executed.

## Documentation

The documentation about code structure and extraction algorithm can be found [here](./docs/Index.md).

## Testing

> This utility is tested using [pytest](https://github.com/pytest-dev/pytest). 

Documentation about the tests can be found [here](./docs/Testing.md). Follow the below commands to run tests on your local system.
1. Install dev-dependencies.
    ```sh
    $ pipenv install --dev
    ```
2. Run tests using `Makefile`.
    ```sh
    $ make test
    ```
3. Run test coverage.
    ```sh
    $ make cov
    ```

## Bugs/Assumptions
-  The utility is built based on the assumption that, there might be empty spaces either in Location or Nature column or both. If there are empty value in any other columns the utility may fail to extract incidents.
- The utility assumes there are only five columns (Datetime, Incident Number, Location, Nature and Incident ORI) for each incident. If that is changed, the utility may fail to extract incidents.