SHELL=/bin/bash
LINT_PATHS=frasa_env/ *.py

mypy:
	mypy ${LINT_PATHS}

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --output-format=full
	# exit-zero treats all errors as warnings.
	ruff check ${LINT_PATHS} --exit-zero --output-format=concise

format:
	# Sort imports
	ruff check --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	ruff check --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format mypy lint

run:
	python enjoy_sbx.py \
		--algo crossq \
		--env frasa-standup-v0 \
		--gym-packages frasa_env \
		--folder logs/ \
		--load-best \
		--exp-id 0

train:
	python train_sbx.py \
		--algo crossq \
		--env frasa-standup-v0 \
		--conf hyperparams/crossq.yml

.PHONY: lint format check-codestyle commit-checks
