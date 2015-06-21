// ctc/ctc-loss.h

// hcq

#ifndef KALDI_CTC_CTC_LOSS_H_
#define KALDI_CTC_CTC_LOSS_H_

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"

namespace kaldi {
namespace nnet1 {

class CTCLoss {
public:
  CTCLoss(int blank_num)
    : blank_(blank_num), frames_(0), loss_(0.0)
  { }
  ~CTCLoss() { }

  /// Evaluate connectionist temporal classification (CTC) errors from labels
  void Eval(const CuMatrixBase<BaseFloat> &log_net_out,
            const std::vector<int32> &target,
            CuMatrix<BaseFloat> *diff);
  
  /// Generate string with error report
  std::string Report();

private:
  /// Evaluate CTC errors on host matrix 
  void eval_on_host(const MatrixBase<BaseFloat> &log_net_out_host,
                    const std::vector<int32> &target,
                    Matrix<BaseFloat> *diff_host);

private:
  int blank_;
  
  int32 frames_;
  double loss_;

};

} // namespace nnet1
} // namespace kaldi

#endif // KALDI_CTC_CTC_LOSS_H_
