#!/bin/bash

## Begin configuration section
use_cmu_tool=false

if $use_cmu_tool; then
  train_tool=/home/hcq/hcq/kaldi/eesen-master/src/nnetbin/nnet-ctc-train
else
  train_tool=ctc-train-perutt
fi

start_epoch_num=52
max_iters=120

learn_rate=0.0001
momentum=0.9

norm_vars=true

verbose=1
## End configuration section

echo "$0 $@"

[ -f path.sh ] && . ./path.sh

. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <data-tr> <data-cv> <exp-dir>"
  echo " e.g.: $0 data/train data/dev exp/train_phn"
fi

data_tr=$1
data_cv=$2
dir=$3

mkdir -p $dir/log $dir/nnet

for f in $data_tr/feats.scp $data_cv/feats.scp $dir/targets.tr.ark $dir/targets.cv.ark $dir/nnet.proto; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 2;
done

## setup up features
cat $data_tr/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
cat $data_cv/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/cv.scp

feats_tr="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$dir/train.scp ark:- |"
feats_cv="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$dir/cv.scp ark:- |"

# add delta
feats_tr="$feats_tr add-deltas --delta-order=2 ark:- ark:- |"
feats_cv="$feats_cv add-deltas --delta-order=2 ark:- ark:- |"

## end of feature setup

## set up labels
labels_tr="ark:$dir/targets.tr.ark"
labels_cv="ark:$dir/targets.cv.ark"
##

# initialize model
if [ ! -f $dir/nnet/nnet.iter0 ]; then
  echo "Initializing model as $dir/nnet/nnet.iter0"
  nnet-initialize --binary=true $dir/nnet.proto $dir/nnet/nnet.iter0 >& $dir/log/initialize_model.log || exit 13;
fi

cur_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
echo "TRAINING STARTS [$cur_time]"
cvacc=0
for iter in $(seq $start_epoch_num $max_iters); do
  cvacc_prev=$cvacc
  echo -n "EPOCH $iter RUNNING ... "

  # train
  if $use_cmu_tool; then
    $train_tool --learn-rate=$learn_rate --momentum=$momentum \
      --verbose=$verbose \
      "$feats_tr" "$labels_tr" $dir/nnet/nnet-cmu.iter$[iter-1] $dir/nnet/nnet-cmu.iter${iter} \
      >& $dir/log/tr.iter$iter.log
  else 
    $train_tool --learn-rate=$learn_rate --momentum=$momentum \
      --verbose=$verbose \
      --blank-num=0 \
      "$feats_tr" "$labels_tr" $dir/nnet/nnet.iter$[iter-1] $dir/nnet/nnet.iter$iter \
      >& $dir/log/tr.iter$iter.log
  fi

  end_time=`date | awk '{print $6 "-" $2 "-" $3 " " $4}'`
  echo -n "ENDS [$end_time]: "

  tracc=$(cat $dir/log/tr.iter${iter}.log | grep "TOKEN_ACCURACY" | tail -n 1 | awk '{ acc=$3; gsub("%", "", acc); print acc; }')
  echo -n "lrate $(printf "%.6g" $learn_rate), TRAIN ACCURACY $(printf "%.4f" $tracc)%, "

  # validation
    $train_tool --learn-rate=$learn_rate --momentum=$momentum \
      --cross-validate=true \
      --verbose=$verbose \
      --blank-num=0 \
      "$feats_cv" "$labels_cv" $dir/nnet/nnet.iter${iter} \
      >& $dir/log/cv.iter${iter}.log
  cvacc=$(cat $dir/log/cv.iter${iter}.log | grep "TOKEN_ACCURACY" | tail -n 1 | awk '{ acc=$3; gsub("%", "", acc); print acc; }')
  echo "VALIDATION ACCURACY $(printf "%.4f" $cvacc)%"

  # stopping criterion
  rel_impr=$(bc <<< "($cvacc-$cvacc_prev)")
  echo "relative improvement in iter ${iter} $(printf "%.4f" $rel_impr)"

done
