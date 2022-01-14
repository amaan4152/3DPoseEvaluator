PROG=src/evaluate.py

.PHONY: 
	all pretty clean

all: 
	python3 $(PROG) 

pretty:
	black src/*
	flake8 --ignore=E,W src/*


