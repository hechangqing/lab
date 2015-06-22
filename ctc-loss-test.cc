// ctc/ctc-loss-test.cc

// hcq

#include "ctc/ctc-loss.h"
#include "base/kaldi-types.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet1 {
  
  /*
   * Helper functions
   */  
  template<typename Real>
  void ReadCuMatrixFromString(const std::string& s, CuMatrix<Real>* m) {
    std::istringstream is(s + "\n");
    m->Read(is, false); // false for ascii
  }
  /*
   */

  void UnitTestCTCLossUnity() {
    // prepare input
    CuMatrix<BaseFloat> nnet_out;
    ReadCuMatrixFromString("[ 0.1 0.7 0.1 0.1;\
                              0.1 0.1 0.7 0.1;\
                              0.1 0.1 0.1 0.7 ] ", &nnet_out);
    nnet_out.ApplyLog();
    // prepare target labels
    int tgts[] = { 1, 2, 3 };
    std::vector<int> targets(tgts, tgts + sizeof(tgts)/sizeof(tgts[0]));
    // store ctc errors
    CuMatrix<BaseFloat> obj_diff;

    // calculate ctc errors
    CTCLoss ctc(0);  // 0 for blank
    ctc.Eval(nnet_out, targets, &obj_diff);
    // prepare thruth errors
    CuMatrix<BaseFloat> obj_diff_truth;
    ReadCuMatrixFromString("[ 0.1 -0.3 0.1 0.1;\
                              0.1 0.1 -0.3 0.1;\
                              0.1 0.1 0.1 -0.3 ] ", &obj_diff_truth);
  
    KALDI_LOG << "log forward variables:\n" << ctc.forward_variables_;
    KALDI_LOG << "log backward variables:\n" << ctc.backward_variables_;
    KALDI_LOG << "calculate  errors:\n" << obj_diff << std::endl;    
    KALDI_LOG << "truth      errors:\n" << obj_diff_truth << std::endl;    
    KALDI_LOG << ctc.Report();
    AssertEqual(obj_diff, obj_diff_truth);
  }
} // namespace nnet1
} // namespace kaldi

int main()
{
  using namespace kaldi;
  using namespace kaldi::nnet1;

  UnitTestCTCLossUnity();
  KALDI_LOG << "Tests succeeded.";
  return 0;
}
