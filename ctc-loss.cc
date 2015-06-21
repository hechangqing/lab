// ctc/ctc-loss.cc

// hcq

#include "ctc/ctc-loss.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

void CTCLoss::Eval(const CuMatrixBase<BaseFloat> &log_net_out,
                   const std::vector<int32> &target,
                   CuMatrix<BaseFloat> *diff)
{
  // download from GPU
  log_net_out_host_.Resize(log_net_out.NumRows(), log_net_out.NumCols());
  log_net_out.CopyToMat(&log_net_out_host_);

  // calculate CTC errors
  eval_on_host(log_net_out_host_, target, &diff_host_);

  diff->Resize(log_net_out.NumRows(), log_net_out.NumCols());
  // -> GPU
  *diff = diff_host_;
}

void eval_on_host(const MatrixBase<BaseFloat> &log_net_out,
                  const std::vector<int32> &target,
                  Matrix<BaseFloat> *diff)
{
  KALDI_ASSERT(blank_ >= 0);
  diff->Resize(log_net_out.NumRows(), log_net_out.NumCols());

  total_time_ = log_net_out.NumRows();
  // check required time > total time
  int required_time = target.size();
  int32 old_label = -1;
  for (size_t i = 0; i != target.size(); i++) {
    if (old_label == target[i]) {
      required_time++;
    }
    old_label = target[i];
  }
  if (total_time_ < required_time) {
    KALDI_ERR << "required time > total time; this should not happen because"
      " this should have been checked before calling this function";
  }
  total_segments_ = target.size() * 2 + 1;

  // calculate the forward variables
  forward_variables_.Resize(total_time_, total_segments_,
                            Log<BaseFloat>::logZero);
  forward_variables_(0, 0) = log_net_out(0, blank_);
  if (total_segments_ > 1) {
    forward_variables_(0, 1) = log_net_out(0, target[0]);
  }
  for (int t = 1; t < total_time_; t++) {
    SubVector<BaseFloat> log_acts(log_net_out, t);
    SubVector<BaseFloat> old_fvars(forward_variables_, t-1);
    SubVector<BaseFloat> fvars(forward_variables_, t);
    std::pair<int, int> this_range = segment_range(t);
    for (int s = this_range.first; s != this_range.second; s++) {
      BaseFloat fv = Log<BaseFloat>::logZero;
      // s odd (label output)
      if (s & 1) {
        int label_index = s / 2;
        int label_num = target[label_index];
        fv = Log<BaseFloat>::log_add(old_fvars(s), old_fvars(s-1));
        if (s > 1 && (label_num != target[label_index-1])) {
          fv = Log<BaseFloat>::log_add(fv, old_fvars(s-2));
        }
        fv = Log<BaseFloat>::log_multiply(fv, log_acts(label_num));
      } else { // s even (blank output)
        fv = old_fvars(s);
        if (s) {
          fv = Log<BaseFloat>::log_add(fv, old_fvars(s-1));
        }
        fv = Log<BaseFloat>::log_multiply(fv, log_acts(blank_));
      }
      fvars(s) = fv;
    } // for (int s)
  } // for (int t)
  
  SubVector<BaseFloat> last_fvars(forward_variables_, total_time_-1);
  BaseFloat log_prob = last_fvars(last_fvars.Dim() - 1);
  if (total_segments_ > 1) {
    log_prob = Log<BaseFloat>::log_add(log_prob, 
                                       last_fvars(last_fvars.Dim() - 2));
  }
  KALDI_ASSERT(log_prob <= 0 && 
      "sequence has log probability " + str(log_prob));

  // calculate the backward variables
  backward_variables_.Resize(total_time_, total_segments_, 
      Log<BaseFloat>::logZero);
  SubVector<BaseFloat> last_bvars(backward_variables_, total_time_-1);
  last_bvars(last_bvars.Dim() - 1) = Log<BaseFloat>::safe_log(1);
  if (total_segments_ > 1) {
    last_bvars(last_bvars.Dim() - 2) = Log<BaseFloat>::safe_log(1);
  }
  // loop over time, calculating back ward variables recursively
  for (int t = total_time_ - 2; t >= 0; t--) {
    SubVector<BaseFloat> old_log_acts(log_net_out, t+1);
    SubVector<BaseFloat> old_bvars(backward_variables_, t+1);
    SubVector<BaseFloat> bvars(backward_variables_, t);
    pair<int, int> this_range = segment_range(t);
    for (int s = this_range.first; s != this_range.second; s++) {
      BaseFloat bv;
      // s odd (label output)
      if (s & 1) {
        int label_index = s / 2;
        int label_num = target[label_index];
        bv = Log<BaseFloat>::log_add(
            Log<BaseFloat>::log_multiply(old_bvars(s), old_log_acts(label_num)),
            Log<BaseFloat>::log_multiply(old_bvars(s+1), old_log_acts(blank_)));
        if (s < total_segments_ - 2) {
          int next_label_num = target[label_index + 1];
          if (label_num != next_label_num) {
            bv = Log<BaseFloat>::log_add(bv,
                Log<BaseFloat>::log_multiply(old_bvars(s+2),
                                             old_log_acts(next_label_num)));
          }
        }
      } else { // s even (blank output)
        bv = Log<BaseFloat>::log_multiply(old_bvars(s), old_log_acts(blank_));
        if (s < total_segments_ - 1) {
          bv = Log<BaseFloat>::log_add(bv,
              Log<BaseFloat>::log_multiply(old_bvars(s+1),
                                           old_log_acts(target[s/2])));
        }
      }
      bvars(s) = bv;
    } // for (int s)
  } // for (int t)

  // inject the training errors
  de_dy_terms_.resize(log_net_out.NumCols());
  for (int time = 0; time < total_time_; time++) {
    fill(de_dy_terms_, Log<BaseFloat>::logZero);
    SubVector<BaseFloat> fvars(forward_variables_, time);
    SubVector<BaseFloat> bvars(backward_variables_, time);
    for (int s = 0; s < total_segments_; s++) {
      // k = blank_ for even s, target label for odd s
      int k = (s&1) ? target[s/2] : blank_;
      de_dy_terms_[k] = Log<BaseFloat>::log_add(de_dy_terms_[k],
          Log<BaseFloat>::log_multiply(fvars(s), bvars(s)));
    }
    for (size_t i = 0; i < de_dy_terms_.size(); i++) {
      diff(time, i) = Log<BaseFloat>::safe_exp(
          Log<BaseFloat>::log_subtract(log_net_out(time, i),
            Log<BaseFloat>::log_divide(de_dy_terms_[i], log_prob)));
    }
  }
}



} // namespace nnet1
} // namespace kaldi
