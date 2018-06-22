#!/bin/sh

python3.5 ProcessPipeline.py "london" "preproc"
python3.5 ProcessPipeline.py "london" "home_sample"
python3.5 ProcessPipeline.py "london" "home_full"
python3.5 ProcessPipeline.py "london" "networks"
python3.5 ProcessPipeline.py "london" "opt_dbscan"

