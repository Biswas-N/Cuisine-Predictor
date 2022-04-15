from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from nltk.stem import WordNetLemmatizer
from joblib import dump
from pathlib import Path
import re
import sys
import os

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


CF_MODEL_NAME = "cf.joblib"
NF_MODEL_NAME = "nf.joblib"
LE_ENCODER_NAME = "le.joblib"


def fit_dump_cf(models_folder: Path, data_file: Path, le: LabelEncoder):
    """Reads Yummly.json data and create a LinearSVC model to
    predict cuisine based on given ingredients

    Parameters
    ----------
    models_folder   : Folder to dump fitted model using joblib
    data_file       : Yummly.json data file path
    """

    yummly_df = load_raw_data(data_file)

    y = le.fit_transform(yummly_df["cuisine"])
    X = yummly_df.drop(["cuisine"], axis=1)

    sys.stderr.write(f"fit_dump_cf:\tBuilding cuisine predictor pipeline...\n")
    preprocessor = ColumnTransformer(
        transformers=[
            ('vectorizer',
             TfidfVectorizer(ngram_range=(1, 1), stop_words="english"),
             "ingredients")])

    clf_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', CalibratedClassifierCV(LinearSVC(C=0.9, penalty='l2')))
    ])

    sys.stderr.write(f"fit_dump_cf:\tFitting model...\n")
    clf_pipe.fit(X, y)

    model_path = os.path.join(models_folder, CF_MODEL_NAME)
    dump(clf_pipe, model_path)

    sys.stderr.write(f"fit_dump_cf:\tModel fitted and saved to {model_path}\n")


def fit_dump_nf(models_folder: Path, data_file: Path, le: LabelEncoder):
    """Reads Yummly.json data and create a kNN model to predict
    N similar recipes based on given ingredients

    Parameters
    ----------
    models_folder   : Folder to dump fitted model using joblib
    data_file       : Yummly.json data file path
    """

    yummly_df = load_raw_data(data_file)

    y = le.fit_transform(yummly_df["cuisine"])
    X = yummly_df.drop(["cuisine"], axis=1)

    sys.stderr.write(
        f"fit_dump_nf:\tBuilding neighbor predictor pipeline...\n")
    preprocessor = ColumnTransformer(
        transformers=[
            ('vectorizer',
             TfidfVectorizer(ngram_range=(1, 1), stop_words="english"),
             "ingredients")])

    knn_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', KNeighborsClassifier(n_neighbors=14, metric="cosine"))
    ])

    sys.stderr.write(f"fit_dump_nf:\tFitting model...\n")
    knn_pipe.fit(X, y)

    model_path = os.path.join(models_folder, NF_MODEL_NAME)
    dump(knn_pipe, model_path)

    sys.stderr.write(f"fit_dump_nf:\tModel fitted and saved to {model_path}\n")


def fit_dump_le(models_folder: Path, data_file: Path):
    """Reads Yummly.json data and creates a LabelEncoder

    Parameters
    ----------
    models_folder   : Folder to dump fitted model using joblib
    data_file       : Yummly.json data file path
    """

    yummly_df = load_raw_data(data_file)

    sys.stderr.write(f"fit_dump_le:\tFitting Label Encoder...\n")
    le = LabelEncoder()
    le.fit(yummly_df["cuisine"])

    model_path = os.path.join(models_folder, LE_ENCODER_NAME)
    dump(le, model_path)

    sys.stderr.write(f"fit_dump_nf:\tModel fitted and saved to {model_path}\n")


def normalize_ingreds(x: list[str]) -> str:
    """Pre-process and clean the raw ingredients from data file

    Parameters
    ----------
    data_file       : Yummly.json data file path
    """

    # Removed any actionable verbs from the ingredient
    skip_verbs = [
        "crushed", "crumbles", "ground", "minced", "chopped",
        "sliced", "grilled", "boneless", "skinless"]
    def remove_verbs(x): return re.sub(r"|".join(skip_verbs), '', x)
    ingreds = list(map(remove_verbs, x))

    # Change the ingredients into their base form (eg. eggs -> egg)
    lemmatizer = WordNetLemmatizer()
    ingreds = [" ".join([lemmatizer.lemmatize(j)
                         for j in i.lower().split(" ")])
               for i in ingreds]

    # Remove any non character or spaces from the ingredient
    # along with replacing multi-spaces with single space, so
    # that it can be replaced with single underscore ('_')
    ingreds = [re.sub("[^A-Za-z ]", "", i) for i in ingreds]
    ingreds = [re.sub(" +", " ", i) for i in ingreds]
    ingreds = [i.strip().replace(" ", "_") for i in ingreds]

    return ",".join(ingreds)


def load_raw_data(data_file: Path) -> pd.DataFrame:
    """Loads the raw data file into a pandas DataFrame

    Parameters
    ----------
    data_file       : Yummly.json data file path
    """

    sys.stderr.write(f"load_raw_data:\tLoading raw data from {data_file}...\n")
    yummly_df = pd.read_json(data_file)

    sys.stderr.write(f"load_raw_data:\tNormalizing Yummly data...\n")
    yummly_df["ingredients"] = yummly_df["ingredients"].map(normalize_ingreds)
    yummly_df = yummly_df[~yummly_df.duplicated(
        ["cuisine", "ingredients"], keep="first")]

    sys.stderr.write(
        f"load_raw_data:\tRaw data loaded into Dataframe ({yummly_df.shape})\n")
    return yummly_df
