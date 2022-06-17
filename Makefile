PROG=src/main.py
DATA=data/GAIT_noexo_00.csv

.PHONY: 
	all init pretty clean

all:
ifeq ($(MODEL),VIBE)
	@bash scripts/set_device.sh "/root/VIBE"
else ifeq ($(MODEL),GAST)
	@bash scripts/set_device.sh "/root/GAST-Net-3DPoseEstimation"
endif
ifeq ($(EVAL),True)
	python3 $(PROG) -d $(DATA) -v $(VIDEO) -m $(MODEL) --start $(START) --end $(END) --eval
else ifeq ($(ANIMATE),True)
	python3 $(PROG) -d $(DATA) -v $(VIDEO) -m $(MODEL) --start $(START) --end $(END) --animate
else
	python3 $(PROG) -d $(DATA) -v $(VIDEO) -m $(MODEL) --start $(START) --end $(END)
endif

# make sure to have `black` and `flake8` installed to prettify all python files
pretty:
	@black src/*.py
	@flake8 --ignore=E,W src/*.py

clean: 
	@rm -rf output/*
