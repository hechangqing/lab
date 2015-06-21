// ctc/log-test.cc

// hcq

#include "base/kaldi-types.h"
#include "Log.hpp"
#include "util/common-utils.h"

#define T BaseFloat
#define log(x) Log<T>::safe_log(x)
#define exp(x) Log<T>::safe_exp(x)
#define add(a,b) Log<T>::log_add(a,b)
#define sub(a,b) Log<T>::log_subtract(a,b)
#define mul(a,b) Log<T>::log_multiply(a,b)
#define div(a,b) Log<T>::log_divide(a,b)

namespace kaldi {
namespace nnet1 {

  void UnitTestLog() {
    AssertEqual(exp(mul(log(0.1), log(0.2))), 0.1 * 0.2);
    T a = 0.00000001;
    T b = 0.00000002;
    AssertEqual(exp(mul(log(a), log(b))), a * b);
    AssertEqual(exp(mul(log(0), log(0.2))), 0 * 0.2);
  }

} // namespace nnet1
} // namespace kaldi

int main()
{
  using namespace kaldi;
  using namespace kaldi::nnet1;  
  // unit-tests:
  UnitTestLog();
  
  KALDI_LOG << "Tests succeeded.";
  return 0;
}
