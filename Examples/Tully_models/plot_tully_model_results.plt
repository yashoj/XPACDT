#!/usr/local/bin/gnuplot

# Simple gnuplot script to plot Tully model results.
# This outputs an eps file named 'result_tully_model_*.eps' where * can be A, B or C.

# Setting output to eps file.
set terminal postscript eps enhanced color font ",24"

# List of all models to be plotted.
model_type_list = "A B C"

set xlabel "Initial momentum [a.u.]"
set ylabel "Probability"
set yrange [-0.1:1.1]


# Plot probabilities for all models in the list
do for [m in model_type_list]{
    set output "result_tully_model_".m.".eps" 
    plot 'results_model_'.m.'/all_p_state_0_refl_adiab.dat' u 1:2:3 w yerrorlines title 'Reflection in state 0', \
         'results_model_'.m.'/all_p_state_1_refl_adiab.dat' u 1:2:3 w yerrorlines title 'Reflection in state 1', \
         'results_model_'.m.'/all_p_state_0_transm_adiab.dat' u 1:2:3 w yerrorlines title 'Transmission in state 0', \
         'results_model_'.m.'/all_p_state_1_transm_adiab.dat' u 1:2:3 w yerrorlines title 'Transmission in state 1'
}

