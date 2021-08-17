code := jambot tests working az_*

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
	@func azure functionapp publish jambot-app --build-native-deps

.PHONY : run-app-local
run-app-local:  ## run app for local testing
	@func host start

.PHONY : testfunc
testfunc:  ## Test trigger azure function running on localhost
	@curl --request POST -H "Content-Type:application/json" --data '{"input":""}' http://localhost:7071/admin/functions/az_RetrainModel

.PHONY : reqs
reqs:  ## make requirements.txt for function app
	@poetry export -f requirements.txt --output requirements.txt --without-hashes

.PHONY : test
test: ## run tests
	@poetry run pytest

help: ## show this help message
	@## https://gist.github.com/prwhite/8168133#gistcomment-1716694
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)" | sort