install:
	PYTHONWARNINGS=ignore poetry install

build:
	poetry build

install-global:
	pip install dist/*.whl

clean:
	rm -rf dist build *.egg-info

pyright-off:
	sed -i '' -E 's/("typeCheckingMode": ")[^"]+/\1off/' pyrightconfig.json; \
	echo "☁️  Pyright mode set to OFF"

pyright-basic:
	sed -i '' -E 's/("typeCheckingMode": ")[^"]+/\1basic/' pyrightconfig.json; \
	echo "🔍 Pyright mode set to BASIC"

pyright-strict:
	sed -i '' -E 's/("typeCheckingMode": ")[^"]+/\1strict/' pyrightconfig.json; \
	echo "⚠️  Pyright mode set to STRICT"
