import os
from pathlib import Path
from joblib import load
import pytest
import numpy as np

from project2 import model_builders


@pytest.fixture(scope="module")
def tests_root():
    tests_folder = Path(__file__).parent.resolve()
    return tests_folder


@pytest.fixture()
def dummy_raw_data_file(tests_root):    
    dummy_file = os.path.join(tests_root, "assets", "yummly_test.json")
    return dummy_file


def test_load_raw_data(dummy_raw_data_file):
    test_df = model_builders.load_raw_data(dummy_raw_data_file)

    assert test_df.shape == (145,3)
    assert (test_df.columns == np.array(['id', 'cuisine', 'ingredients'])).all()


def test_normalize_ingreds():
    inputs = [
        "crushed cashew",
        " minced      chicken ",
        "  black     pepper *"
    ]

    got = model_builders.normalize_ingreds(inputs)

    assert got == 'cashew,chicken,black_pepper'


@pytest.fixture(scope="module")
def dummy_models_folder(tests_root):    
    dummy_models_fldr = os.path.join(tests_root,  "assets", "models")
    Path(dummy_models_fldr).mkdir(parents=True, exist_ok=True)
    
    yield dummy_models_fldr

    os.rmdir(dummy_models_fldr)


def test_fit_dump_le(dummy_models_folder, dummy_raw_data_file):
    model_builders.fit_dump_le(dummy_models_folder, dummy_raw_data_file)

    model_file = os.path.join(
        dummy_models_folder, model_builders.LE_ENCODER_NAME)
    
    assert os.path.exists(model_file) == True

    os.remove(model_file)
    assert os.path.exists(model_file) == False


model_dump_testcases = [
    (model_builders.fit_dump_cf, model_builders.CF_MODEL_NAME),
    (model_builders.fit_dump_nf, model_builders.NF_MODEL_NAME)
]

@pytest.mark.parametrize("test_func, model_name", model_dump_testcases)
def test_fit_dump_finder_models(
    test_func, model_name, dummy_models_folder, dummy_raw_data_file
):
    model_builders.fit_dump_le(dummy_models_folder, dummy_raw_data_file)
    dummy_le_path = os.path.join(
        dummy_models_folder, model_builders.LE_ENCODER_NAME)
    dummy_le = load(dummy_le_path)

    test_func(dummy_models_folder, dummy_raw_data_file, dummy_le)

    model_file = os.path.join(
        dummy_models_folder, model_name)
    
    assert os.path.exists(model_file) == True

    os.remove(model_file)
    assert os.path.exists(model_file) == False

    os.remove(dummy_le_path)
    assert os.path.exists(dummy_le_path) == False
