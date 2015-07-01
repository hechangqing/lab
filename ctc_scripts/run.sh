#!/bin/bash

#
# Copyright 2013 Bagher BabaAli,
#           2014 Brno University of Technology (Author: Karel Vesely)
#
# TIMIT, description of the database:
# http://perso.limsi.fr/lamel/TIMIT_NISTIR4930.pdf
#
# Hon and Lee paper on TIMIT, 1988, introduces mapping to 48 training phonemes, 
# then re-mapping to 39 phonemes for scoring:
# http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci
#

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

stage=3

feats_nj=10
train_nj=20
decode_nj=5

echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================

timit=/home/pcc/disk3/hcq/timit

if [ $stage -le 1 ]; then
  local/timit_data_prep.sh $timit || exit 1
  
  local/timit_prepare_dict.sh
  
  # Caution below: we remove optional silence by setting "--sil-prob 0.0",
  # in TIMIT the silence appears also as a word in the dictionary and is scored.
  utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
   data/local/dict "sil" data/local/lang_tmp data/lang
  
  local/timit_format_data.sh
fi

echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set          "
echo ============================================================================

# Now make MFCC features.
mfccdir=mfcc
if [ $stage -le 2 ]; then
  for x in train dev test; do 
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
  done
fi

echo ============================================================================
echo "                             CTC Training                              "
echo ============================================================================
# Specify network structure and generate the network topology
input_feat_dim=39  # 13-dimensional MFCC wiht deltas
lstm_layer_num=1   # number of LSTM layers
lstm_cell_dim=256  # number of memroy cells in every LSTM layer

target_num=40 # the number of phones + 1 (the blank)

dir=exp/train_phn_tgt${target_num}_l${lstm_layer_num}_c${lstm_cell_dim}
mkdir -p $dir

# network topology
ctc_scripts/model_topo.py --input-feat-dim $input_feat_dim --lstm-layer-num $lstm_layer_num \
  --lstm-cell-dim $lstm_cell_dim --target-num $target_num > $dir/nnet.proto || exit 1;

# label sequences; simply convert phones into their label indices
 # map 48 phones to 39 phones
ctc_scripts/map-to-39.py data/train/text conf/phones.60-48-39.map data/train/text.39
ctc_scripts/map-to-39.py data/dev/text conf/phones.60-48-39.map data/dev/text.39
 # code phones
cut -d ' ' -f 2- data/train/text.39 | tr ' ' '\n' | awk 'NF' | sort | uniq > $dir/phones.39.list
cat $dir/phones.39.list | awk 'NF' | awk '{print $0 " " NR}' > $dir/phones.39.txt
 # make targets
text-to-target data/train/text.39 $dir/phones.39.txt ark:$dir/targets.tr.ark
text-to-target data/dev/text.39 $dir/phones.39.txt ark:$dir/targets.cv.ark
 # do some checking
for targets in "$dir/targets.tr.ark" "$dir/targets.cv.ark"; do
    echo "check" $targets "... "
    tmp=$(mktemp --tmpdir=. tmp.XXX)
    copy-int-vector ark:$targets ark,t:- | cut -d ' ' -f 2- | tr ' ' '\n' | awk 'NF' | sort -n | uniq > $tmp
    min_phn=$(head -n 1 $tmp)
    max_phn=$(tail -n 1 $tmp)
    echo -n "LOG: min_phn_num" $min_phn
    echo " max_phn_num" $max_phn
    ! [ $max_phn -le $[target_num-1] ] && echo "ERROR TARGETS" && exit 19
    rm $tmp
done

# train the network with CTC.
echo "train"
ctc_scripts/train_ctc.sh data/train data/dev $dir || echo "train ctc error" $? && exit 22;
echo "end"

exit 0;
