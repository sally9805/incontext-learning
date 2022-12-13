#!/bin/bash
set -x

. ./consts.sh

#bash run_pretrain.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms6" 8
#
#bash run_pretrain.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms10" 8
#
#bash run_pretrain.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms14" 8
#


bash run_incontext.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms6" 6000 8

bash run_incontext.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms10" 6000 8

bash run_incontext.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms14" 6000 8

bash run_pretrain.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms18" 8

bash run_incontext.sh "GINC_trans0.1_start10.0_nsymbols100_nvalues10_nslots10_vic0.9_nhmms18" 6000 8
