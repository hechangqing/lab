#include "base/kaldi-common.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

#include <numeric> // accumulate

void split(const string &s, const string &delim, vector<string> *ret)
{
    ret->clear();
    string::size_type index = 0, last = 0;
    do {
        last = s.find_first_not_of(delim, index);
        index = s.find_first_of(delim, last);
        if (index > last) {
            ret->push_back(s.substr(last, index-last));
        }
    } while (index != string::npos);
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using std::ifstream;
  using std::multimap;
  using std::map;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Transform text to targets file in ctc train\n"
        "Usage: text-to-target [options] <text> <phone-syms> <targets-wspecifier>\n"
        "e.g.: \n"
        "  text-to-target text phones.txt ark:targets.ark\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string text_filename = po.GetArg(1),
        phones_symtab_filename = po.GetArg(2),
        targets_wspecifier = po.GetArg(3);
    
    fst::SymbolTable *phones_symtab = NULL;
    {
        std::ifstream is(phones_symtab_filename.c_str());
        phones_symtab = fst::SymbolTable::ReadText(is, phones_symtab_filename);
        if (!phones_symtab || phones_symtab->NumSymbols() == 0) {
            KALDI_ERR << "Error opening symbol table file " << phones_symtab_filename;
        }
    }
    
    Int32VectorWriter targets_writer(targets_wspecifier);
    
    ifstream is(text_filename.c_str());
    if (!is) {
      KALDI_ERR << "Error opening text file " << text_filename;
    } 
    string record;
    vector<string> split_record;
    while (getline(is, record)) {
      split(record, " \t\n", &split_record);
      if (split_record.size() < 2) {
        KALDI_ERR << "Error transcript file " << text_filename
                  << " line " << record;
      }
      const string &key = split_record[0];
      vector<int32> targets;
      for (size_t i = 1; i < split_record.size(); i++) {
        int32 phn = phones_symtab->Find(split_record[i]);
        if (phn == fst::SymbolTable::kNoSymbol) {
          KALDI_ERR << "Error can not find phone " << split_record[i]
                    << " in phone symbol table file " << phones_symtab_filename;
        }
        targets.push_back(phn);
      }
      targets_writer.Write(key, targets);
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
} 

