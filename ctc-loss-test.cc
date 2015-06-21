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

    int tgts[] = { 1, 2, 3 };
    std::vector<int> targets(tgts, tgts + sizeof(tgts)/sizeof(tgts[0]));

    CuMatrix<BaseFloat> obj_diff;

    CTCLoss ctc(0);
    ctc.Eval(nnet_out, targets, &obj_diff);
    
    std::cout << ctc.Report();
  }
} // namespace nnet1
} // namespace kaldi

int main()
{
  using namespace kaldi;
  using namespace kaldi::nnet1;

  UnitTestCTCLossUnity();
  return 0;
}
