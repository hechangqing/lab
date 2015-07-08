// ctc/ctc-loss.h

// hcq

#ifndef KALDI_CTC_CTC_LOSS_H_
#define KALDI_CTC_CTC_LOSS_H_

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"
#include <utility>

namespace kaldi {
namespace nnet1 {

class CTCLoss {
public:
  CTCLoss(int blank_num, int report_step = 100)
    : blank_(blank_num), total_time_(0), total_segments_(0),
      frames_(0), sequences_num_(0), ref_num_(0), error_num_(0.0), frames_progress_(0),
      sequences_progress_(0), ref_num_progress_(0), error_num_progress_(0.0),
      obj_progress_(0.0), report_step_(report_step)
  { KALDI_ASSERT(report_step > 0); }
  ~CTCLoss() { }

  /// Evaluate connectionist temporal classification (CTC) errors from labels
  void Eval(const CuMatrixBase<BaseFloat> &log_net_out,
            const std::vector<int32> &target,
            CuMatrix<BaseFloat> *diff);
  
  /// the net_out can be log scale net out or just net out,
  ///   because we just need the relative value
  void ErrorRate(const CuMatrixBase<BaseFloat> &net_out,
                 const std::vector<int32> &label,
                 double *error_rate,
                 std::vector<int32> *hyp);
  /// Generate string with error report
  std::string Report();

public:
  /// Evaluate CTC errors on host matrix 
  void eval_on_host(const MatrixBase<BaseFloat> &log_net_out_host,
                    const std::vector<int32> &target,
                    Matrix<BaseFloat> *diff_host);
  
  std::pair<int, int> segment_range(int time) const;
public:
  int blank_;
 
  int total_time_;
  int total_segments_;
  std::vector<BaseFloat> de_dy_terms_;
  Matrix<BaseFloat> forward_variables_;
  Matrix<BaseFloat> backward_variables_;
  Matrix<BaseFloat> log_net_out_host_;
  Matrix<BaseFloat> diff_host_;

  int32 frames_;              // total number of frames
  int32 sequences_num_;       // total number of sequences
  int32 ref_num_;             // total number of tokens in label sequences
  double error_num_;          // total number of errors (edit distance between hyp and ref)
  
  int32 frames_progress_;
  int32 sequences_progress_;  // registry for the number of sequences
  int32 ref_num_progress_;
  double error_num_progress_;
  
  double obj_progress_;       // registry for the log optimization objective

  int32 report_step_;         // report obj and accuracy every so many sequences/utterances

};

} // namespace nnet1
} // namespace kaldi

#endif // KALDI_CTC_CTC_LOSS_H_
