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
 
  void UnitTestCTCLossUnity_test_error(
        const std::string &nnet_out_str, 
        const std::vector<int> &targets, 
        const std::string &obj_diff_truth_str) {
    // prepare input
    CuMatrix<BaseFloat> nnet_out;
    ReadCuMatrixFromString(nnet_out_str, &nnet_out);
    nnet_out.ApplyLog();
    
    // prepare target labels
    //int tgts[] = { 1, 2, 3 };
    //std::vector<int> targets(tgts, tgts + sizeof(tgts)/sizeof(tgts[0]));
    
    // store ctc errors
    CuMatrix<BaseFloat> obj_diff;

    // calculate ctc errors
    CTCLoss ctc(0);  // 0 for blank
    ctc.Eval(nnet_out, targets, &obj_diff);
    // prepare thruth errors
    CuMatrix<BaseFloat> obj_diff_truth;
    ReadCuMatrixFromString(obj_diff_truth_str, &obj_diff_truth);
  
    KALDI_LOG << "log forward variables:\n" << ctc.forward_variables_;
    KALDI_LOG << "log backward variables:\n" << ctc.backward_variables_;
    KALDI_LOG << "calculate  errors:\n" << obj_diff << std::endl;    
    KALDI_LOG << "truth      errors:\n" << obj_diff_truth << std::endl;    
    KALDI_LOG << ctc.Report();
    AssertEqual(obj_diff, obj_diff_truth);
  }
  
  void UnitTestCTCLossUnity() {
    // 1
    {
      std::string nnet_out_str = "[ 0.1 0.7 0.1 0.1;\
                                    0.1 0.1 0.7 0.1;\
                                    0.1 0.1 0.1 0.7 ] ";
      int tgts[] = { 1, 2, 3 };
      std::vector<int> targets(tgts, tgts + sizeof(tgts)/sizeof(tgts[0]));
      std::string obj_diff_truth_str = 
                             "[ 0.1 -0.3 0.1 0.1;\
                                0.1 0.1 -0.3 0.1;\
                                0.1 0.1 0.1 -0.3 ] ";
      UnitTestCTCLossUnity_test_error(nnet_out_str, targets, obj_diff_truth_str);
    }
    // 2
    {
      std::string nnet_out_str = "[ 0.1 0.7 0.1 0.1;\
                              0.1 0.7 0.1 0.1;\
                              0.1 0.7 0.1 0.1;\
                              0.1 0.7 0.1 0.1;\
                              0.1 0.1 0.7 0.1;\
                              0.1 0.1 0.7 0.1;\
                              0.1 0.1 0.7 0.1;\
                              0.1 0.1 0.1 0.7;\
                              0.1 0.1 0.1 0.7;\
                              0.1 0.1 0.1 0.7;\
                              0.1 0.1 0.1 0.7 ] ";
      int tgts[] = { 1, 2, 3 };
      std::vector<int> targets(tgts, tgts + sizeof(tgts)/sizeof(tgts[0]));
      std::string obj_diff_truth_str =
              " [ -0.0406171 -0.159383 0.1 0.1;\
                   0.0766188 -0.275116 0.0984968 0.1;\
                   0.0695175 -0.254635 0.0851201 0.0999975;\
                   -0.0117153 -0.0761384 -0.0121165 0.0999703;\
                   -0.00573041 -3.58224e-05 -0.0939889 0.0997552;\
                   0.0584813 0.0901614 -0.238804 0.0901614;\
                   -0.00573038 0.0997552 -0.093989 -3.58298e-05;\
                   -0.0117153 0.0999703 -0.0121165 -0.0761385;\
                   0.0695175 0.0999975 0.0851201 -0.254635;\
                   0.0766188 0.1 0.0984968 -0.275116;\
                   -0.0406171 0.1 0.1 -0.159383 ] ";
      UnitTestCTCLossUnity_test_error(nnet_out_str, targets, obj_diff_truth_str);
    }
    // 3
    {
      std::string nnet_out_str = "[ 0.2 0.2 0.2 0.2 0.2;\
                                    0.2 0.2 0.2 0.2 0.2;\
                                    0.2 0.2 0.2 0.2 0.2;\
                                    0.2 0.2 0.2 0.2 0.2;\
                                    0.2 0.2 0.2 0.2 0.2;\
                                    0.2 0.2 0.2 0.2 0.2;\
                                    0.2 0.2 0.2 0.2 0.2;\
                                    0.2 0.2 0.2 0.2 0.2;\
                                    0.2 0.2 0.2 0.2 0.2;\
                                    0.2 0.2 0.2 0.2 0.2;\
                                    0.2 0.2 0.2 0.2 0.2 ] ";
      int tgts[] = { 3, 1, 2, 4 };
      std::vector<int> targets(tgts, tgts + sizeof(tgts)/sizeof(tgts[0]));
      std::string obj_diff_truth_str =
                   "[ -0.266666 0.2 0.2 -0.333334 0.2;\
                      -0.14359 0.076923 0.2 -0.333333 0.2;\
                      -0.124942 -0.0871795 0.181352 -0.169231 0.2;\
                      -0.123699 -0.191609 0.121678 -0.00512818 0.198757;\
                      -0.123699 -0.191609 0.0172495 0.10676 0.191298;\
                      -0.123698 -0.104584 -0.104584 0.166434 0.166434;\
                      -0.123699 0.0172495 -0.191609 0.191298 0.10676;\
                      -0.123699 0.121678 -0.191609 0.198757 -0.00512815;\
                      -0.124942 0.181352 -0.0871795 0.2 -0.169231;\
                      -0.14359 0.2 0.076923 0.2 -0.333333;\
                      -0.266666 0.2 0.2 0.2 -0.333334 ]";
      UnitTestCTCLossUnity_test_error(nnet_out_str, targets, obj_diff_truth_str);
    }

  }

} // namespace nnet1
} // namespace kaldi

int main()
{
  using namespace kaldi;
  using namespace kaldi::nnet1;
 
  try {
    for (int loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
      if (loop == 0)
        CuDevice::Instantiate().SelectGpuId("no");
      else
        CuDevice::Instantiate().SelectGpuId("yes");
#endif

      UnitTestCTCLossUnity();
      
      if (loop == 0)
        KALDI_LOG << "Tests without GPU use succeeded.";
      else
        KALDI_LOG << "Tests with GPU use (if available) succeeded.";
    }
#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch (std::exception e) {
    std::cout << e.what() << std::endl;
    return 1;
  }
}
