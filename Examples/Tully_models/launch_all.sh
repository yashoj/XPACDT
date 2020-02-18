#!/bin/bash

# Run all Tully model examples.
# The script 'run_tully_model.sh' takes the arguments <model_type>, <p_start>,
# <p_step> and <p_stop> respectively where p is the initial momentum.

# Please note that each run with one p value takes about 30-45 mins,
# so running one model can take several hours depending upon the step size.
# If running only a particular model is desired, then comment out the others.

./run_tully_model.sh  model_A  0.0  1.0  30.0  &> model_A.out
./run_tully_model.sh  model_B  9.0  1.0  65.0  &> model_B.out
./run_tully_model.sh  model_C  5.0  1.0  35.0  &> model_C.out

