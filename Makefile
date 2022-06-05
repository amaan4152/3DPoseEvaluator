PROG=src/main.py
DATA=data/GAIT_noexo_00.csv

.PHONY: 
	all init pretty clean

all: init
ifeq ($(EVAL),True)
	python3 $(PROG) -v $(VIDEO) -d $(DATA) -m $(MODEL) --eval
else ifeq ($(ANIMATE),True)
	python3 $(PROG) -v $(VIDEO) -d $(DATA) -m $(MODEL) --animate
else
	python3 $(PROG) -v $(VIDEO) -d $(DATA) -m $(MODEL)
endif

init: 
	bash scripts/set_device.sh

pretty:
	@black src/*.py
	@flake8 --ignore=E,W src/*.py

clean: 
	@rm -rf output/*
