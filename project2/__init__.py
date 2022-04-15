import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
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
    _, neighbors = nf[-1].kneighbors(X=inp_vec, n_neighbors=n)

    yummly_df = load_raw_data(DATA_FILE)
    X = yummly_df.drop(["cuisine"], axis=1)

    neighbors_list = []
    for neighbor_dish_id in neighbors[0]:
        neighbor_ingreds = X.iat[neighbor_dish_id, 1]

        neighbors_list.append([neighbor_dish_id, neighbor_ingreds])

    neighbors_df = pd.DataFrame(neighbors_list, columns=["id", "ingredients"])

    similarities = cosine_similarity(nf[0].transform(neighbors_df), inp_vec)
    neighbors_df["similarities"] = similarities[:, 0]

    return list(
        neighbors_df.loc[:, ["id", "similarities"]].itertuples(
            index=False, name=None)
    )

