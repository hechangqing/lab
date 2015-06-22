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
  CTCLoss(int blank_num)
    : blank_(blank_num), total_time_(0), total_segments_(0), frames_(0), loss_(0.0)
  { }
  ~CTCLoss() { }

  /// Evaluate connectionist temporal classification (CTC) errors from labels
  void Eval(const CuMatrixBase<BaseFloat> &log_net_out,
            const std::vector<int32> &target,
            CuMatrix<BaseFloat> *diff);
  
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

  int32 frames_;
  double loss_;
};

} // namespace nnet1
} // namespace kaldi

#endif // KALDI_CTC_CTC_LOSS_H_
