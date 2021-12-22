from espnet2.tasks.asr import ASRTask
from espnet2.bin.asr_inference import Speech2Text
import numpy as np
import torch


def pcm2float(sig, dtype='float32'):
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")
    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


model = Speech2Text(
        asr_train_config="../ref/mdl/exp/asr_train_asr_transformer2_ddp_raw_bpe/config.yaml",
        asr_model_file='../ref/mdl/exp/asr_train_asr_transformer2_ddp_raw_bpe/valid.acc.ave_10best.pth',
        lm_train_config=None,
        lm_file=None,
        token_type=None,
        bpemodel=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        maxlenratio=0.0,
        minlenratio=0.0,
        dtype='float32',
        beam_size=20,
        ctc_weight=0.3,
        lm_weight=1.0,
        penalty=0.0,
        nbest=1,
    )

preprocess_fn=ASRTask.build_preprocess_fn(model.asr_train_args, False)


def inference(frames):
    frame=pcm2float(frames.get_array_of_samples())
    tens=preprocess_fn('1',{'speech':frame}) #input : (uid,dict)-> output : dict{'speech':array}
    output=model(**{'speech':torch.from_numpy(tens['speech'])}) #input : dict{'speech':Tensor,'speech_lengths':Tensor}
    return output[0][0]