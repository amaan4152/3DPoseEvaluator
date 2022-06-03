PROG=src/main.py
DATA=data/GAIT_noexo_00.csv

.PHONY: 
	all pretty clean

all:
ifeq ($(EVAL),True)
	python3 $(PROG) -v $(VIDEO) -m $(MODEL) --eval
else ifeq ($(ANIMATE),True)
	python3 $(PROG) -v $(VIDEO) -m $(MODEL) --animate
else
	python3 $(PROG) -v $(VIDEO) -m $(MODEL)
endif

pretty:
	@black src/*.py
	@flake8 --ignore=E,W src/*.py

clean: 
	@rm -rf output/*
