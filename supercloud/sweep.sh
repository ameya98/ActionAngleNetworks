#!/bin/bash

SWEEP="sweeps/harmonic_motion/performance_vs_parameters"
CONFIGDIR="action_angle_networks/configs/harmonic_motion"

for CONFIG in "action_angle_flow" "euler_angle_flow" "neural_ode" "hamiltonian_neural_network"
do
    SWEEP=$SWEEP CONFIG="$CONFIGDIR/$CONFIG.py" LLsub ./submit.sh [1,20,2]
done
