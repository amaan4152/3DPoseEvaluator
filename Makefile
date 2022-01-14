PROG=src/evaluate.py
VIDEO=video/GAIT_00.mp4
DATA=data/noexo_00.csv

.PHONY: 
	all pretty clean

all: 
	python3 $(PROG) -v $(VIDEO) -d $(DATA) -t GAIT -m $(MODEL) --start $(START) --end $(END)

pretty:
	black src/*
	flake8 --ignore=E,W src/*


