#!/bin/bash

export script=$HOME/Documents/Codes/green-mbpt-exp/python/migrate.py

cd GW
python $script --old_input_file input.h5 \
  --old_integral_path df_int \
  --new_input_file input.h5 \
  --new_integral_path df_int \
  --version 0.2.4

cd ../GW_X2C
python $script --old_input_file input.h5 \
  --old_integral_path df_hf_int \
  --new_input_file input.h5 \
  --new_integral_path df_hf_int \
  --version 0.2.4

cd ../HF
python $script --old_input_file input.h5 \
  --old_integral_path df_int df_hf_int \
  --new_input_file input.h5 \
  --new_integral_path df_int df_hf_int \
  --version 0.2.4

cd ../HF_X2C
python $script --old_input_file input.h5 \
  --old_integral_path df_hf_int \
  --new_input_file input.h5 \
  --new_integral_path df_hf_int \
  --version 0.2.4

cd ..

