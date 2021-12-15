#!/bin/bash
DUMMY := dummy
DUMMY := dummy
DUMMY := dummy
DUMMY := dummy
DUMMY := dummy
DUMMY := dummy
DUMMY := dummy
DUMMY := dummy
DUMMY := dummy
DUMMY := dummy
DUMMY := dummy

SHELL := /bin/bash
utils := @poetry run python -m scripts.utils
code := jambot tests working az_*

include .vscode/.env

# Azure commands
# func azure functionapp fetch-app-settings jambot-app

# az functionapp config appsettings set --name jambot-app --resource-group jambot-app --settings AzureWebJobs.OneHour.Disabled=true

.PHONY : format
format:  ## autopep, isort, flake
	@poetry run autopep8 --recursive --in-place $(code)
	@poetry run isort $(code)
	@poetry run flake8 $(code)

.PHONY : app
app:  ## push jambot app to azure
	@if ! docker info >/dev/null 2>&1; then\
		echo "Starting Docker";\
		open /Applications/Docker.app;\
	fi
	@func azure functionapp publish jambot-app --python --build-native-deps

.PHONY : run-app-local
run-app-local:  ## run app for local testing
	@poetry run func host start

.PHONY : testfunc
testfunc:  ## Test trigger azure function running on localhost
	@curl --request POST -H "Content-Type:application/json" --data '{"input":""}' http://localhost:7071/admin/functions/az_RetrainModel

.PHONY : reqs
reqs:  ## make requirements.txt for function app
	@poetry export -f requirements.txt --output requirements.txt --without-hashes

.PHONY : test
test:  ## run tests
	@poetry run pytest --cov=jambot --cov-report=xml --cov-report=html
	@~/codecov -f coverage.xml -t ${CODECOV_TOKEN}

.PHONY : showcov
showcov:  ## show coverage report
	@poetry run coverage report

.PHONY : creds
creds:  ## re-encrypt credentials
	$(utils) --encrypt_creds

.PHONY : fit_models
fit_models:  ## fit models for last 3 days, upload to azure
	$(utils) --fit_models

.PHONY : mlflow
mlflow: ## show mlflow UI in browser
	@open http://127.0.0.1:5000
	@poetry run mlflow ui --backend-store-uri ${MLFLOW_CONN}

.PHONY : codecount
codecount:  ## show lines of code
	@pygount --suffix=py --format=summary jambot

.PHONY : init
init:  ## install steps for M1 Mac
	# maybe don't actually need scipy, comes with numpy
	@brew install openblas # for scipy/pandas/sklearn
	@export OPENBLAS=$(brew --prefix openblas)
	@export CFLAGS="-falign-functions=8 ${CFLAGS}" (for scipy maybe)
	@brew install llvm@11 # for shap/numba
	@export LLVM_CONFIG=/opt/homebrew/Cellar/llvm@11/11.1.0_3/bin/llvm-config
	@brew install cmake
	@brew install libomp
	@brew install ta-lib  # not actually used, but python ta-lib is just a wrapper around this
	@brew install unixodbc  # pyodbc
	@brew install libjpeg  # matplotlib
	@export PATH="/opt/homebrew/opt/llvm/bin:$PATH" # matplotlib needs to use llvm@13 not 11

	# for numpy 1.20.3 (for numba, for shap)
	@export MACOSX_DEPLOYMENT_TARGET=12.0
	@poetry run pip install numpy==1.20.3 --no-use-pep517

	# for azure app
	@brew install azuer-cli
	@brew install azure-functions-core-tools@4

help: ## show this help message
	@## https://gist.github.com/prwhite/8168133#gistcomment-1716694
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)" | sort