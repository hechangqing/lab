// ctc/helper.h

// hcq

#ifndef _KALDI_CTC_HELPER_H
#define _KALDI_CTC_HELPER_H

#include <string>
#include <sstream>

template<class T> 
static std::string str(const T &t)
{
  std::stringstream ss;
  ss << t;
  return ss.str();
}

#endif // _KALDI_CTC_HELPER_H
