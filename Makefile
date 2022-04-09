run:
	pipenv run python project2.py --N 5 \
								  --ingredient paprika \
                                  --ingredient banana \
                                  --ingredient "rice krispies"

test:
	pipenv run python -m pytest

cov:
	pipenv run python -m pytest --cov=project2

cov-report:
	pipenv run python -m pytest --cov=project2 --cov-report=html

lint:
	pipenv run python -m autopep8 --in-place --aggressive --recursive .
