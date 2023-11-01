#!/bin/bash

# Define a list of n_steps values
n_steps_values="1 2 3 4 5"

# Loop through the values and run your Python script
for n_steps in $n_steps_values
do
	echo Pitch shifting using value: $n_steps
    #python3 myscript.py --n_steps $n_steps
done
