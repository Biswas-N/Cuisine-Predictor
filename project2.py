import argparse
import sys
import json

import project2
from project2.model_utils import load_models


def main(args: argparse.Namespace):
    """Predicts cusine and closest recipes using ML models (LinearSVC and
    k-Neighbors Classifier)

    Parameters
    ----------
    args    : Parsed command line args
    """

    # Getting the Cusine Finder (cf) and Neighbors Finder (nf) models
    cf, nf, le = load_models()

    cuisine_name, score = project2.get_cusine_prediction(
        cf, le, args.ingredient)
    closest = project2.closest_recipes(
        nf, le, args.ingredient, args.N)

    json_raw_output = {}
    json_raw_output["cuisine"] = cuisine_name
    json_raw_output["score"] = round(score, 2)
    json_raw_output["closest"] = [{"id": str(c[0]), "score": round(c[1], 2)}
        for c in closest]

    json_formatted_output = json.dumps(json_raw_output, indent=2)

    sys.stdout.write(f"{json_formatted_output}\n")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Predicts cuisine and recommends recipes based on given ingredients')
    arg_parser.add_argument(
        "--N",
        required=True,
        type=int,
        help="<Required> number of similar recipes to show")
    arg_parser.add_argument(
        "--ingredient",
        required=True,
        action="append",
        help="<Required> recipe ingredient")

    args = arg_parser.parse_args()

    try:
        sys.stderr.write("\n")

        main(args)
    except Exception as e:
        sys.stderr.write(str(e) + "\n")
        sys.stderr.write(str(e.with_traceback()) + "\n")
