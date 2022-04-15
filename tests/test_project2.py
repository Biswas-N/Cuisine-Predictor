import project2
from project2.model_utils import load_models


def test_say_hello():
    ingredients = [
        "chili powder",
        "crushed red pepper flakes",
        "garlic powder",
        "sea salt",
        "ground cumin",
        "onion powder",
        "dried oregano",
        "ground black pepper",
        "paprika"
    ]

    cf, nf, le = load_models()

    cuisine_name, score = project2.get_cusine_prediction(
        cf, le, ingredients)
    closest = project2.closest_recipes(
        nf, le, ingredients, 5)

    assert cuisine_name == "mexican"
    assert score > 0
    assert len(closest) == 5
