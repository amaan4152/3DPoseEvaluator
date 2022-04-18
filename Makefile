PROG=src/evaluate.py
DATA=data/GAIT_noexo_00.csv

.PHONY: 
	all pretty clean

all: 
	@python3 $(PROG) -v $(VIDEO) -d $(DATA) -t GAIT -m $(MODEL) --no_eval --start $(START) --end $(END)

pretty:
	@black src/*.py
	@flake8 --ignore=E,W src/*.py


