#!/bin/bash

values=('cQQ1' 'cQQ8' 'cQt1' 'cQt8' 'ctt1')

for value in "${values[@]}"
do
    higher="EFT_${value}_higher"
    lower="EFT_${value}_lower"
    higher_NP2="EFT_${value}_higher_NP2"
    lower_NP2="EFT_${value}_lower_NP2"
    higher_NP4="EFT_${value}_higher_NP4"
    lower_NP4="EFT_${value}_lower_NP4"


    echo "Running program.py with arg1=${higher}"
    python create_root_auto.py "$higher"

    echo "Running program.py with arg1=${lower}"
    python create_root_auto.py "$lower"

    echo "Running program.py with arg1=${higher_NP2}"
    python create_root_auto.py "$higher_NP2"

    echo "Running program.py with arg1=${lower_NP2}"
    python create_root_auto.py "$lower_NP2"

    echo "Running program.py with arg1=${higher_NP4}"
    python create_root_auto.py "$higher_NP4"

    echo "Running program.py with arg1=${lower_NP4}"
    python create_root_auto.py "$lower_NP4"

done