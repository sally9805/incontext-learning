#!/bin/bash
set -x

. ./consts.sh

bash rnn.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms10_most_diverse"

bash rnn.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms10_most_diverse" 10

#bash rnn.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms14" 6
#
#bash rnn.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms14" 8
#
#bash rnn.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms14" 10