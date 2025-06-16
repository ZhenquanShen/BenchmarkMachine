from .config import mimic_cfg
from data_utils.file_helper import FileHelper as fh

class Model_Config:
    batch_size = 64
    max_epochs = 80
    dec_seq_length_max = 47
    inc_seq_length_max = 256
    using_gpu = True
    print_every = 100