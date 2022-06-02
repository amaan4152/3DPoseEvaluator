PROG=src/main.py
DATA=data/GAIT_noexo_00.csv

.PHONY: 
	all pretty clean

all:
ifeq ($(EVAL),True)
	python3 $(PROG) -v $(VIDEO) -d $(DATA) -t GAIT -m $(MODEL) --start $(START) --end $(END) --eval
else ifeq ($(ANIMATE),True)
	python3 $(PROG) -v $(VIDEO) -d $(DATA) -t GAIT -m $(MODEL) --start $(START) --end $(END) --animate
else
	python3 $(PROG) -v $(VIDEO) -d $(DATA) -t GAIT -m $(MODEL) --start $(START) --end $(END)
endif

pretty:
	@black src/*.py
	@flake8 --ignore=E,W src/*.py

clean: 
	@rm -rf output/*
