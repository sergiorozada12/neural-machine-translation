REGEX = '(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
NAME = 'Helsinki-NLP/opus-mt-ca-es'
BATCH_SIZE = 32

# Alternative backend fbgemm
QUANTIZATION_BACKEND = 'fbgemm'

MODEL_CONFIG = {
    'default': False,
    'num_beams': 1,
    'top_k': 50,
    'quantize': True,
    'early_stopping': True,
    'do_sample': False,
    'repetition_penalty': 2.0,
    'max_time': 4.0,
}