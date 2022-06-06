PROG=src/main.py

.PHONY: 
	all init pretty clean

all:
ifeq ($(MODEL),VIBE)
	@bash scripts/set_device.sh "/root/VIBE"
else ifeq ($(MODEL),GAST)
	@bash scripts/set_device.sh "/root/GAST-Net-3DPoseEstimation"
endif
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
