#!/bin/bash

# Define a list of n_steps values
n_steps_values="-1 -2 -3 -4 -5 1 2 3 4 5"
DEST_PATH=/home/thanos/Documents/Thesis/labels/TheBeatles_shifted
DIRECTORY=/home/thanos/Documents/Thesis/labels/TheBeatles_lab

# Loop through the values and run your Python script
for n_steps in $n_steps_values
do
	echo Pitch shifting using value: $n_steps

  python3 ../src/pitch_shift_labels.py --directory $DIRECTORY --dest_dir $DEST_PATH/shifted_$n_steps --n_steps $n_steps --pool
done
