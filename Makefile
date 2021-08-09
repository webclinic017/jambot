code := jambot tests working

.PHONY : format
format:  ## autopep, isort, flake
	@poetry run autopep8 --recursive --in-place $(code)
	@poetry run isort $(code)
	@poetry run flake8 $(code)

.PHONY : app
app:  ## push jambot app to azure
	@func azure functionapp publish jambot-app

.PHONY : test
test: ## run tests
	@poetry run pytest

help: ## show this help message
	@## https://gist.github.com/prwhite/8168133#gistcomment-1716694
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)" | sort