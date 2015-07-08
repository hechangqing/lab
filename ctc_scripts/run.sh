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

stage=7

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
echo "                             CTC Training                                 "
echo ============================================================================
# Specify network structure and generate the network topology
input_feat_dim=39  # 13-dimensional MFCC wiht deltas and delta-deltas
lstm_layer_num=1   # number of LSTM layers
lstm_cell_dim=256  # number of memroy cells in every LSTM layer

target_num=40 # the number of phones + 1 (the blank)

dir=exp/train_phn_tgt${target_num}_l${lstm_layer_num}_c${lstm_cell_dim}
mkdir -p $dir

if [ $stage -le 3 ]; then
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
  echo "training ctc..."
  ctc_scripts/train_ctc.sh --start-epoch-num 1 --max-iters 95 data/train data/dev $dir || echo "train ctc error" $? && exit 22;
  echo "train ctc finished"
fi

echo ============================================================================
echo "                      CTC Testing (Phone Recognition)                     "
echo ============================================================================

test_dir=$dir/phn-recog
mkdir -p $test_dir $test_dir/log

if [ $stage -le 4 ]; then
    echo "testing..."
    # prepare ground truth label sequences
    ctc_scripts/map-to-39.py data/test/text conf/phones.60-48-39.map data/test/text.39
     # make targets
    text-to-target data/test/text.39 $dir/phones.39.txt ark:$dir/targets.test.ark
     # do some checking. make sure that the phones(int value) in test ark file match the network
     tmp=$(mktemp --tmpdir=. tmp.XXX)
     copy-int-vector ark:$dir/targets.test.ark ark,t:- | cut -d ' ' -f 2- | tr ' ' '\n' | awk 'NF' | sort -n | uniq > $tmp
     min_phn=$(head -n 1 $tmp)
     max_phn=$(tail -n 1 $tmp)
     echo -n "LOG: min_phn_num" $min_phn
     echo " max_phn_num" $max_phn
     ! [ $max_phn -le $[target_num-1] ] && echo "ERROR TEST TARGETS" && exit 27
     rm $tmp
    # setup feature
    norm_vars=true
    cat data/test/feats.scp > $dir/test.scp
    feats_test="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:data/test/utt2spk scp:data/test/cmvn.scp scp:$dir/test.scp ark:- |"
    feats_test="$feats_test add-deltas --delta-order=2 ark:- ark:- |"
    # end of feature setup
    # setup labels
    labels_test="ark:$dir/targets.test.ark"
    # end of setup labels
    echo "TESTING STARTS"
    # we just do cross-validate on the test set to test the net
    for((i=20;i<=95;i++)); do
      ctc-train-perutt --cross-validate=true --verbose=1 --blank-num=0 "$feats_test" "$labels_test" $dir/nnet/nnet.iter$i \
        >& $test_dir/log/test.iter$i.log
      acc=$(cat $test_dir/log/test.iter${i}.log | grep "TOKEN_ACCURACY" | tail -n 1 | awk '{ acc=$3; gsub("%", "", acc); print acc; }')
      echo "MODEL $i TEST ACCURACY $(printf "%.4f" $acc)%"
    done
fi

## get the best model
phn_recog_result="$test_dir/result.log"
echo -n "" > $phn_recog_result
for((i=20;i<=95;i++)); do
  acc=$(cat $test_dir/log/test.iter${i}.log | grep "TOKEN_ACCURACY" | tail -n 1 | awk '{ acc=$3; gsub("%", "", acc); print acc; }')
  echo "MODEL nnet.iter$i TEST ACCURACY $(printf "%.4f" $acc)%"
  echo "nnet.iter$i $acc" >> $phn_recog_result
done
best=$(sort -n -k 2 $phn_recog_result | tail -n 1)
best_net=$(echo $best | awk '{ mdl=$1; print mdl; }')
best_acc=$(echo $best | awk '{ acc=$2; print acc; }')
best_ler=$(echo $best | awk '{ ler=100-$2; print ler; }')
echo "BEST MODEL $best_net LER: $best_ler%  TOKEN_ACC: $best_acc%"

echo ============================================================================
echo "                      CTC Testing (End-to-End Recognition)                "
echo ============================================================================
if [ $stage -le 5 ]; then 
  # prepare decoding fst TLG.fst in data-ctc/lang_test_bg
  ctc_scripts/timit_compile_dict_token.sh $dir data/train

  # Compute the occurrence counts of labels in the label sequences. These counts will be used to derive prior probabilities of
  # the labels.
  copy-int-vector ark:$dir/targets.tr.ark ark,t:- | awk '{line=$0; gsub(" ", " 0 ", line); print line " 0";}' | \
  analyze-counts --verbose=1 --binary=false ark:- $dir/label.counts >& $dir/log/compute_label_counts.log || exit 1
fi 

decode_dir=$dir/decode-test
mkdir -p $decode_dir $decode_dir/log

if [ $stage -le 6 ]; then
    # setup feature
    norm_vars=true
    cat data/test/feats.scp > $dir/test.scp
    feats_test="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:data/test/utt2spk scp:data/test/cmvn.scp scp:$dir/test.scp ark:- |"
    feats_test="$feats_test add-deltas --delta-order=2 ark:- ark:- |"
    # end of feature setup
    nnet-forward --class-frame-counts=$dir/label.counts --apply-log=true --no-softmax=false $dir/nnet/$best_net "$feats_test" ark:- | \
    ctc-decode-faster --beam=15 --max-active=7000 --acoustic-scale=0.9 --word-symbol-table=data-ctc/lang_test_bg/words.txt \
    --allow-partial=true data-ctc/lang_test_bg/TLG.fst ark:- ark,t:$decode_dir/trans >& $decode_dir/log/decode.log

fi

 cat data/test/text.39 > $decode_dir/text

 cat $decode_dir/trans | utils/int2sym.pl -f 2- data-ctc/lang_test_bg/words.txt | \
   compute-wer --text --mode=present ark:$decode_dir/text ark,p:-


exit 0;
