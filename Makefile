install:
	PYTHONWARNINGS=ignore poetry install

build:
	poetry build

install-global:
	pip install dist/*.whl

clean:
	rm -rf dist build *.egg-info
