#!/bin/bash

if [ $# -ne 2 ]; then
  echo "usage: ctc_scripts/timit_compile_dict_token.sh <exp-dir> <data-train-dir>"
  exit 1;
fi

exp=$1
train_dir=$2
dir=data-ctc
lang=data-ctc/lang
tmpdir=data-ctc/lang_tmp
mkdir -p $dir $tmpdir $lang

cp $exp/phones.39.txt $lang/.

# make lexicon from phones
awk '{ print $1 "\t" $1; }' $exp/phones.39.txt > $lang/lexicon.txt

# Add probalilities to lexicon entries. There is in fact no point of doing this here since all the entries have 1.0.
# But utils/make_lexicon_fst.pl requires a probilistic version, so we just leave it as it is.
perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $lang/lexicon.txt > $tmpdir/lexiconp.txt || exit 1;

# Add disambiguation symbols to the lexicon. This is necessary for determinizing the composition of L.fst and G.fst.
# Without these symbols, determinization will fail.
ndisambig=`utils/add_lex_disambig.pl --pron-probs $tmpdir/lexiconp.txt $tmpdir/lexiconp_disambig.txt`
ndisambig=$[$ndisambig+1]
echo $ndisambig

( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) > $tmpdir/disambig.list

# Get the full list of CTC tokens used in FST. These tokens include <eps>, the blank <blk>, the actual labels (here is phonemes),
# and the disambiguation symbols.
awk '{print $1;}' $exp/phones.39.txt > $tmpdir/units.list
(echo '<eps>'; echo '<blk>';) | cat - $tmpdir/units.list $tmpdir/disambig.list | awk '{print $1 " " (NR-1)}' > $lang/tokens.txt

# Compile the tokens into FST
ctc_scripts/ctc_token_fst.py $lang/tokens.txt | fstcompile --isymbols=$lang/tokens.txt --osymbols=$lang/tokens.txt \
  --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $lang/T.fst || exit 1;

# Encode the words with indices. Will be used in lexicon ang languag model FST compiling.
cat $tmpdir/lexiconp.txt | awk '{print $1}' | sort | uniq | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
  }' > $lang/words.txt || exit 1;

# Now compile the lexicon FST. Depending on the size of your lexicon, it may take some time.
token_disambig_symbol=`grep \#0 $lang/tokens.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 $lang/words.txt | awk '{print $2}'`

utils/make_lexicon_fst.pl --pron-probs $tmpdir/lexiconp_disambig.txt 0 "sil" '#'$ndisambig | \
  fstcompile --isymbols=$lang/tokens.txt --osymbols=$lang/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
  fstaddselfloops "echo $token_disambig_symbol |" "echo $word_disambig_symbol |" | \
  fstarcsort --sort_type=olabel > $lang/L.fst || exit 1;

# Create the phone bigram LM
  [ -z "$IRSTLM" ] && \
    echo "LM building won't work without setting the IRSTLM env variable" && exit 1;
  ! which build-lm.sh 2>/dev/null  && \
    echo "IRSTLM does not seem to be installed (build-lm.sh not on your path): " && \
    echo "go to <kaldi-root>/tools and try 'make irstlm_tgt'" && exit 1;

  echo "Preparing language models for testing"
  cut -d' ' -f2- $train_dir/text.39 | sed -e 's:^:<s> :' -e 's:$: </s>:' \
    > $lang/lm_train.text
  build-lm.sh -i $lang/lm_train.text -n 2 -o $tmpdir/lm_phone_bg.ilm.gz

  compile-lm $tmpdir/lm_phone_bg.ilm.gz -t=yes /dev/stdout | \
  grep -v unk | gzip -c > $lang/lm_phone_bg.arpa.gz 

  test=data-ctc/lang_test_bg
  mkdir -p $test
  cp $lang/words.txt $test || exit 1; 

  gunzip -c $lang/lm_phone_bg.arpa.gz | \
    egrep -v '<s> <s>|</s> <s>|</s> </s>' | \
    arpa2fst - | fstprint | \
    utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$test/words.txt \
     --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > $test/G.fst
  fstisstochastic $test/G.fst
 # The output is like:
 # 9.14233e-05 -0.259833
 # we do expect the first of these 2 numbers to be close to zero (the second is
 # nonzero because the backoff weights make the states sum to >1).
 # Because of the <s> fiasco for these particular LMs, the first number is not
 # as close to zero as it could be.
 
 # Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
 # minimized.
 fsttablecompose $lang/L.fst $test/G.fst | fstdeterminizestar --use-log=true | \
   fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst || exit 1;
   fsttablecompose $lang/T.fst $tmpdir/LG.fst > $test/TLG.fst || exit 1;

echo "Composing decoding graph TLG.fst succeeded"
rm -r $tmpdir
