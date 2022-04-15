from joblib import dump, load
from pathlib import Path
import os
import sys

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from project2.model_builders import fit_dump_cf, fit_dump_nf, fit_dump_le
from project2.model_builders import CF_MODEL_NAME, NF_MODEL_NAME
from project2.model_builders import LE_ENCODER_NAME


__MODELS_FOLDER = Path(os.path.join(os.getcwd(), "models")).resolve()
__ASSETS_FOLDER = Path(os.path.join(os.getcwd(), "assets")).resolve()
DATA_FILE = Path(os.path.join(__ASSETS_FOLDER, "yummly.json"))


def load_models() -> tuple[Pipeline, Pipeline, LabelEncoder]:
    sys.stderr.write("Loading models...\n")
    
    if not os.path.exists(__ASSETS_FOLDER) or not os.path.exists(DATA_FILE):
        raise Exception(f"Yummly.json not present in {__ASSETS_FOLDER}")

    if not os.path.exists(__MODELS_FOLDER):
        sys.stderr.write(f"Models folder not found! Creating models folder..\n")
        __MODELS_FOLDER.mkdir(parents=True)

    cf_path = os.path.join(__MODELS_FOLDER, CF_MODEL_NAME) # Cusine Finder Model
    nf_path = os.path.join(__MODELS_FOLDER, NF_MODEL_NAME) # Neighbors Finder Model
    le_path = os.path.join(__MODELS_FOLDER, LE_ENCODER_NAME) # Label Encoder Model

    if not os.path.exists(le_path):
        fit_dump_le(__MODELS_FOLDER, DATA_FILE)
    le = load(le_path)
    
    if not os.path.exists(cf_path):
        fit_dump_cf(__MODELS_FOLDER, DATA_FILE, le)

    if not os.path.exists(nf_path):
        fit_dump_nf(__MODELS_FOLDER, DATA_FILE, le)

    sys.stderr.write(f"\nload_models:\tModels loaded\n\n")
    return load(cf_path), load(nf_path), load(le_path)
