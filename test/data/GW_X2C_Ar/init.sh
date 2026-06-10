#!/bin/bash

# Point GREEN_MBPT_PYTHON at the python/ directory of your green-mbpt checkout,
# or override ScriptDir directly.  Example:
#   GREEN_MBPT_PYTHON=/path/to/green-mbpt/python ./init.sh
: "${ScriptDir:=${GREEN_MBPT_PYTHON:?Set GREEN_MBPT_PYTHON or ScriptDir to the green-mbpt/python directory}}"

export basis="def2-svp"
rm -f -- tmp* cderi*
python3 "$ScriptDir/init_data_df.py" --x2c 2 --a a.dat --atom atom.dat \
  --basis $basis --auxbasis def2-svp-jfit --nk 3 3 1 --xc PBE --keep_cderi true \
  --space_symm true --tr_symm true --df_int 1 --output_path input_full_symm.h5 \
  --use_j2c_eig_decomposition false

rm -f -- tmp* cderi*
python3 "$ScriptDir/init_data_df.py" --x2c 2 --a a.dat --atom atom.dat \
  --basis $basis --auxbasis def2-svp-jfit --nk 3 3 1 --xc PBE --keep_cderi true \
  --space_symm false --tr_symm true --df_int 0 --output_path input_trs_only.h5 \
  --use_j2c_eig_decomposition false

rm -f -- tmp* cderi*
python3 "$ScriptDir/init_data_df.py" --x2c 2 --a a.dat --atom atom.dat \
  --basis $basis --auxbasis def2-svp-jfit --nk 3 3 1 --xc PBE --keep_cderi true \
  --space_symm false --tr_symm false --df_int 0 --output_path input_no_symm.h5 \
  --use_j2c_eig_decomposition false

