# Makefile to create directory structure

.PHONY: all dirs clean

all: dirs

dirs:
	mkdir -p data
	mkdir -p data/splits
	mkdir -p data/participant_folders
	mkdir -p output
	mkdir -p output/au
	mkdir -p output/gaze
	mkdir -p output/gaze/boxplots
	mkdir -p output/au/boxplots

clean:
	rm -rf data output