#!/bin/bash

. ./consts.sh

N_VALUES=8
N_SLOTS=8
N_SYMBOLS=130
TRANS_TEMP=0.1
START_TEMP=10.0
VIC=0.9
OTHER_ARGS=$8
SEED=1111

python generate_data.py \
    --transition_temp $TRANS_TEMP \
    --start_temp $START_TEMP \
    --n_symbols $N_SYMBOLS \
    --n_values $N_VALUES \
    --n_slots $N_SLOTS \
    --value_identity_coeff $VIC \
    --sample_diver_hmm \
    --n_hmms 12 \
    --root "."
#    --list_of_n_hmms "22" \

bash rnn.sh "GINC_trans0.1_start10.0_nsymbols130_nvalues8_nslots8_vic0.9_nhmms12_most_diverse" 10