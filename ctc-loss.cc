// ctc/ctc-loss.cc

// hcq

#include "ctc/ctc-loss.h"
#include "ctc/helper.h"
#include "util/edit-distance.h"
#include "cudamatrix/cu-math.h"
#include "base/kaldi-types.h"
#include "ctc/Log.hpp"
#include <algorithm> 

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

void CTCLoss::eval_on_host(const MatrixBase<BaseFloat> &log_net_out,
                  const std::vector<int32> &target,
                  Matrix<BaseFloat> *diff)
{
  KALDI_ASSERT(blank_ >= 0);
  diff->Resize(log_net_out.NumRows(), log_net_out.NumCols());

  total_time_ = log_net_out.NumRows();
  // check required time > total time
  {
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
  }
  
  total_segments_ = target.size() * 2 + 1;
  
  // calculate the forward variables
  forward_variables_.Resize(total_time_, total_segments_, kUndefined);
  forward_variables_.Set(Log<BaseFloat>::logZero);
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
  
  // std::cout << "log prob " << log_prob << std::endl;
  KALDI_ASSERT(log_prob <= 0);

  // calculate the backward variables
  backward_variables_.Resize(total_time_, total_segments_, kUndefined);
  backward_variables_.Set(Log<BaseFloat>::logZero);
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
    std::pair<int, int> this_range = segment_range(t);
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
    std::fill(de_dy_terms_.begin(), de_dy_terms_.end(),
        Log<BaseFloat>::logZero);
    SubVector<BaseFloat> fvars(forward_variables_, time);
    SubVector<BaseFloat> bvars(backward_variables_, time);
    for (int s = 0; s < total_segments_; s++) {
      // k = blank_ for even s, target label for odd s
      int k = (s&1) ? target[s/2] : blank_;
      de_dy_terms_[k] = Log<BaseFloat>::log_add(de_dy_terms_[k],
          Log<BaseFloat>::log_multiply(fvars(s), bvars(s)));
      //std::cout << "dedy" << k << " " << Log<BaseFloat>::safe_exp(de_dy_terms_[k]) << std::endl;
    }
    for (size_t i = 0; i < de_dy_terms_.size(); i++) {
      (*diff)(time, i) = 
          Log<BaseFloat>::safe_exp(log_net_out(time, i)) -
          Log<BaseFloat>::safe_exp(
              Log<BaseFloat>::log_divide(de_dy_terms_[i], log_prob));
      //std::cout << "net_out " << Log<BaseFloat>::safe_exp(log_net_out(time, i)) << " ";
      //std::cout << "dedy/logprob " << Log<BaseFloat>::safe_exp(Log<BaseFloat>::log_divide(de_dy_terms_[i], log_prob)) << " ";
      //std::cout << "diff" << time << i << " " << (*diff)(time, i) << std::endl;
    }
  }

  // record progress
  obj_progress_ += log_prob;
  sequences_progress_ += 1;
  sequences_num_ += 1;
  frames_progress_ += total_time_;
  frames_ += total_time_;

  // progress reporting
  {
    if (sequences_progress_ >= report_step_) {
      KALDI_VLOG(1) << "After " << sequences_num_ << " sequences ("
                    << frames_/(100.0 * 3600) << "Hr): "
                    << "Obj(log[P(z|x)]) = " << obj_progress_/sequences_progress_
                    << "   TokenAcc = "
                    << 100.0*(1.0-error_num_progress_/ref_num_progress_)
                    << "%";
      sequences_progress_ = 0;
      frames_progress_ = 0;
      obj_progress_ = 0;
      error_num_progress_ = 0;
      ref_num_progress_ = 0;
    }
  }
}

std::pair<int, int> CTCLoss::segment_range(int time) const
{
  int start = std::max(0, total_segments_ - (2 * (total_time_ - time)));
  int end = std::min(total_segments_, 2 * (time + 1));
  end = (start > end ? start : end);
  KALDI_ASSERT(start <= end);
  return std::make_pair(start, end);
  //return std::make_pair(0, total_segments_);
}

void CTCLoss::ErrorRate(const CuMatrixBase<BaseFloat> &net_out,
               const std::vector<int32> &label,
               double *error_rate,
               std::vector<int32> *hyp)
{
  CuArray<int32> maxid(net_out.NumRows());
  net_out.FindRowMaxId(&maxid);

  std::vector<int32> maxid_host(net_out.NumRows());
  maxid.CopyToVec(&maxid_host);

  // remove repetitions and blanks
  int32 i = 1, j = 1;
  int32 dim = maxid_host.size();
  for (; j < dim; j++) {
    if (maxid_host[j] != maxid_host[j-1]) {
      maxid_host[i++] = maxid_host[j];
    }
  }
  hyp->resize(0);
  for (int32 n = 0; n < i; n++) {
    if (maxid_host[n] != blank_) {
      hyp->push_back(maxid_host[n]);
    }
  }
  
  int32 err, ins, del, sub;
  err = LevenshteinEditDistance(label, *hyp, &ins, &del, &sub);
  *error_rate = (100.0 * err) / label.size();
  error_num_ += err;
  ref_num_ += label.size();
  error_num_progress_ += err;
  ref_num_progress_ += label.size();
}

std::string CTCLoss::Report()
{
  std::ostringstream oss;
  oss << "\nTOKEN_ACCURACY >> " << 100.0 * (1.0 - error_num_ / ref_num_)
      << "% <<";
  return oss.str();
}

} // namespace nnet1
} // namespace kaldi
