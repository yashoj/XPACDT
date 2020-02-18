#!/bin/bash

# Run Tully model example.
# This script takes the arguments <model_type>, <p_start>, <p_step> and <p_stop> 
# respectively, where p represents the initial momentum.
# And it outputs files with names in the format '<model_type>_*.dat' where * could be for example 'state_0_refl_adiab'.
# In these output files, the columns are p, value and error of result.

model=$1
input_file=full_tully_model.in

# Start, step and stpp for momentum in the model.
p_start=$2
p_step=$3
p_stop=$4

result_file_list=("state_0_refl_adiab.dat" "state_1_refl_adiab.dat" "state_0_transm_adiab.dat" "state_1_transm_adiab.dat")

# Create output folders and files
out_data_folder=tully_${model}_data
mkdir ${out_data_folder}

result_folder=results_${model}
mkdir ${result_folder}

for f in ${result_file_list[@]}; do
    out_file=${result_folder}/all_p_${f}

    if [ -f ${out_file} ]; then
        rm ${out_file}
    fi

    touch ${out_file}
    # Add header to output file
    echo  "# Result of Tully ${model} for `echo ${f} | sed 's/\.dat//'`." >> ${out_file}
    echo  "# Columns are initial momentum p, value and error of result respectively." >> ${out_file}
done

# Temporary file to change and use as input file.
tmp_file=full_${model}_tmp.in

# Loop over the different initial momentum p.
# The '-w' option gives equal width with leading zeros.
for p in `seq -w ${p_start} ${p_step} ${p_stop}`; do
    echo $'\n'
    echo ${p}

    folder_name=`echo "${out_data_folder}/p_${p}" | sed 's/\./_/g' `
    echo ${folder_name}

    cp ${input_file} ${tmp_file}
    sed -i s/VAR_MODEL/${model}/g  ${tmp_file}
    sed -i s/VAR_MOMENTA/${p}/g  ${tmp_file} 
    sed -i s#VAR_FOLDER#${folder_name}#g  ${tmp_file}

    xpacdt.py -i ${tmp_file}

    # Collect the data in the output files; first column is momenta, then value and error
    for f in ${result_file_list[@]}; do
        out_file=${result_folder}/all_p_${f}

        echo "${p}  `tail -1 ${folder_name}/${f} | awk '{print $2 "  " $3}'`" >> ${out_file}
    done

    rm ${tmp_file}
done

rm -r ${out_data_folder}

