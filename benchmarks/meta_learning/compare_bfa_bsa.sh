#!/usr/bin/bash
for L in 128 256 512; do
  echo 'Biased Flex Attention'
  ./test_biased_flex_attention.py -L $L
  echo 'Biased Scan Attention'
  ./test_biased_scan_attention.py -L $L
  echo ''
done
