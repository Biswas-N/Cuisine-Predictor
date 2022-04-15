run:
	pipenv run python project2.py \
		--N 5 \
		--ingredient "chili powder" \
		--ingredient "crushed red pepper flakes" \
		--ingredient "garlic powder" \
		--ingredient "sea salt" \
		--ingredient "ground cumin" \
		--ingredient "onion powder" \
		--ingredient "dried oregano" \
		--ingredient paprika 

test:
	pipenv run python -m pytest

cov:
	pipenv run python -m pytest --cov=project2

cov-report:
	pipenv run python -m pytest --cov=project2 --cov-report=html

lint:
	pipenv run python -m autopep8 --in-place --aggressive --recursive .
