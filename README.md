# Cuisine Predictor
## Developer: Biswas Nandamuri
Cuisine Predictor is a python based tool wrapping ML models which use Scikit Learn's LinearSVC and kNeighborsClassifier to predict the cuisine and similar dishes from [Yummly catlog](./assets/yummly.json).

> The project's python code follows PEP8 Style Guide

#### Dependencies
* [Scikit-learn](https://scikit-learn.org/stable/) - scikit-learn is a Python module for machine learning built on top of SciPy
* [Pandas](https://github.com/pandas-dev/pandas) - Flexible and powerful data analysis / manipulation library for Python
* [Nltk](https://github.com/nltk/nltk) - NLTK is a platform for building Python programs to work with human language data

#### Dev Dependencies
* [Pytest](https://github.com/pytest-dev/pytest) - Testing framework that supports complex functional testing
* [Pytest-cov](https://github.com/pytest-dev/pytest-cov) - Coverage plugin for pytest
* [autopep8](https://github.com/hhatto/autopep8) - Tool that automatically formats Python code to conform to the PEP 8 style guide
* [Jupyterlab](https://github.com/jupyterlab/jupyterlab) - Browser-based computational environment for python
* [Matplotlib](https://github.com/matplotlib/matplotlib) - Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python
* [Seaborn](https://seaborn.pydata.org/) - Seaborn is a Python data visualization library based on matplotlib


## Run on local system
1. Clone this repository and move into the folder.
    ```sh
    $ git clone https://github.com/Biswas-N/cs5293sp22-project2.git
    $ cd cs5293sp22-project2
    ```
2. Install dependencies using [Pipenv](https://github.com/pypa/pipenv).
    ```sh
    $ pipenv install
    ``` 
3. Run the utility tool
    ```sh
    $ make
    ```
   > Note: Project includes a `Makefile` which has commonly used commands. By running `make` the following command `pipenv run python project2.py --N 5 --ingredient "chili powder" --ingredient "crushed red pepper flakes" --ingredient "garlic powder" --ingredient "sea salt" --ingredient "ground cumin" --ingredient "onion powder" --ingredient "dried oregano" --ingredient paprika` is executed.

   > Note on Model: If pre-fitted models does not exist in `models` folder, this tool creates fitted-models based on `assets/yummly.json` data and stores them in `models` folder using `joblib`. So the first execution may take more time than typical execution times.

## Documentation

The documentation about code structure, model building and prediction algorithm can be found [here](./docs/Index.md).

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
- This tool assumes that the `yummly.json` is present in `assets` folder. So if the data file is not present in the folder, this tool may fail.
- Similarly, this tool assumes there are three keys in each JSON object in `yummly.json` file called `id`, `cuisine` and `ingredients` (should be a list of ingredients). If the data inside `yummly.json` is not as expected, the tool may fail.
- This tool is built using LinearSVC and KNeighborsClassifier, and trained using the given `yummly.json` data. So this tools accuracy is based on the data quality provided and statistical techniques used in the above said models. I tried using the best approaches possible for pre-processing and model fitting in the given time constraints. But there is always more to do, so there might be cases in which the tool can predict less-accurate results.
