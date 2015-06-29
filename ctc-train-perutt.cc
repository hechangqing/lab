// ctc/ctc-train-perutt.cc

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "ctc/ctc-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by Connectionist Temporal Classification (CTC).\n"
        "This version use labels as targets.\n"
        "The updates are done per-utternace, shuffling options are dummy for compatibility reason.\n"
        "\n"
        "Usage:  ctc-train-perutt [options] --blank-num=integer <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " ctc-train-perutt --blank-num=0 scp:feature.scp ark:target.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    int blank_num = -1;
    po.Register("blank-num", &blank_num, "The number that stands for blank in network. >= 0");
    
    int report_step = 100;
    po.Register("report-step", &report_step, "The steps that print report");

    bool binary = true, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/weights (frames)");
    
    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    // Add dummy randomizer options, to make the tool compatible with standard scripts
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    bool randomize = false;
    po.Register("randomize", &randomize, "Dummy option, for compatibility...");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0) || blank_num < 0) {
      if (blank_num < 0) {
        std::cerr << "invalid blank-num " << blank_num << "; blank-num should >= 0" << std::endl;
      }
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader targets_reader(targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    PosteriorRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    CTCLoss ctc_loss(blank_num);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, obj_diff;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    for ( ; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      KALDI_VLOG(3) << "Reading " << utt;
      // check that we have targets
      if (!targets_reader.HasKey(utt)) {
        KALDI_WARN << utt << ", missing targets";
        num_no_tgt_mat++;
        continue;
      }
      // check we have per-frame weights
      if (frame_weights != "" && !weights_reader.HasKey(utt)) {
        KALDI_WARN << utt << ", missing per-frame weights";
        num_other_error++;
        feature_reader.Next();
        continue;
      }
      // get feature / target pair
      Matrix<BaseFloat> mat = feature_reader.Value();
      const std::vector<int32> &targets = targets_reader.Value(utt);
      // get per-frame weights
      Vector<BaseFloat> weights;
      if (frame_weights != "") {
        weights = weights_reader.Value(utt);
      } else { // all per-frame weights are 1.0
        weights.Resize(mat.NumRows());
        weights.Set(1.0);
      }
      // correct small length mismatch ... or drop sentence
      {
        // add lengths to vector
        std::vector<int32> lenght;
        lenght.push_back(mat.NumRows());
        lenght.push_back(weights.Dim());
        // find min, max
        int32 min = *std::min_element(lenght.begin(),lenght.end());
        int32 max = *std::max_element(lenght.begin(),lenght.end());
        // fix or drop ?
        if (max - min < length_tolerance) {
          if(mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
          if(weights.Dim() != min) weights.Resize(min, kCopyData);
        } else {
          KALDI_WARN << utt << ", length mismatch of weights " << weights.Dim()
                     << " and features " << mat.NumRows();
          num_other_error++;
          continue;
        }
      }
      // check features length is enough for targets or drop sentence
      {
        int total_time = mat.NumRows();
        int required_time = targets.size();
        int old_label = -1;
        for (size_t i = 0; i != targets.size(); i++) {
          if (old_label == targets[i]) {
            required_time++;
          }
          old_label = targets[i];
        }
        if (total_time < required_time) {
          KALDI_WARN << utt << ", required time > total time";
          num_other_error++;
          continue;
        }
      }
      // apply optional feature transform
      nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);
 
      // get block of feature/target pairs
      //const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();

      // forward pass
      nnet.Propagate(feats_transf, &nnet_out);
      
      // apply log
      nnet_out.ApplyLog();

      // evaluate objective function
      ctc_loss.Eval(nnet_out, targets, &obj_diff);
      double err = 0.0;
      std::vector<int32> hyp;
      ctc_loss.ErrorRate(nnet_out, targets, &err, &hyp);
      // backward pass
      if (!crossvalidate) {
        // re-scale the gradients
        obj_diff.MulRowsVec(CuVector<BaseFloat>(weights));
        // backpropagate
        nnet.Backpropagate(obj_diff, NULL);
      }

      // 1st minibatch : show what happens in network 
      if (kaldi::g_kaldi_verbose_level >= 1 && total_frames == 0) { // vlog-1
        KALDI_VLOG(1) << "### After " << total_frames << " frames,";
        KALDI_VLOG(1) << nnet.InfoPropagate();
        if (!crossvalidate) {
          KALDI_VLOG(1) << nnet.InfoBackPropagate();
          KALDI_VLOG(1) << nnet.InfoGradient();
        }
      }
      
      // monitor the NN training
      if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
        if ((total_frames/25000) != ((total_frames+feats_transf.NumRows())/25000)) { // print every 25k frames
          KALDI_VLOG(2) << "### After " << total_frames << " frames,";
          KALDI_VLOG(2) << nnet.InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(2) << nnet.InfoGradient();
          }
        }
      }
      
      // report the speed
      num_done++;
      total_frames += feats_transf.NumRows();
      if (num_done % 5000 == 0) {
        double time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << total_frames/time_now
                      << " frames per second.";
#if HAVE_CUDA==1
        // check the GPU is not overheated
        CuDevice::Instantiate().CheckGpuHealth();
#endif
      }
    }
      
    // after last minibatch : show what happens in network 
    if (kaldi::g_kaldi_verbose_level >= 1) { // vlog-1
      KALDI_VLOG(1) << "### After " << total_frames << " frames,";
      KALDI_VLOG(1) << nnet.InfoPropagate();
      if (!crossvalidate) {
        KALDI_VLOG(1) << nnet.InfoBackPropagate();
        KALDI_VLOG(1) << nnet.InfoGradient();
      }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  

    KALDI_LOG << ctc_loss.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
