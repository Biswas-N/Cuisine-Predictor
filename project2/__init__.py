import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from project2.model_builders import normalize_ingreds, load_raw_data
from project2.model_utils import DATA_FILE


def get_cusine_prediction(
    cf: Pipeline, le: LabelEncoder, ingredients: list[str]
) -> tuple[str, float]:
    
    input_df = pd.DataFrame(
        [normalize_ingreds(ingredients)], columns=["ingredients"])

    pred_probs = cf.predict_proba(input_df)

    cuisine, score = sorted(
        [(c, s) for c, s in zip(le.classes_, pred_probs[0])],
        key=lambda x: x[1],
        reverse=True
    )[0]

    return cuisine, score


def closest_recipes(
    nf: Pipeline, le: LabelEncoder, ingredients: list[str], n: int
) -> list[tuple[str, float]]:
    
    input_df = pd.DataFrame(
        [normalize_ingreds(ingredients)], columns=["ingredients"])
    
    inp_vec = nf[0].transform(input_df)
    dists, neighbors = nf[-1].kneighbors(X=inp_vec, n_neighbors=n)

    similars = list((n, 1 - d) 
        for n,d in zip(neighbors[0], dists[0]))
    
    X = load_raw_data(DATA_FILE)

    return list((X.iat[row_id, 0], score) 
        for row_id, score in similars)
