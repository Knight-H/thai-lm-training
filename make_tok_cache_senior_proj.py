import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"

from pathlib import Path
import torch

from torch.utils.data import Dataset, DataLoader
from tokenizers import CharBPETokenizer, Tokenizer, ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import BertNormalizer
# from tokenizers import SentencePieceBPETokenizer

import random
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast , AutoTokenizer,RobertaTokenizerFast, RobertaTokenizer
from filelock import FileLock
import logging
import time
import tqdm
import pickle
from multiprocessing import Pool
from functools import reduce

DATA_PATH = Path("/datadisk")

# DATA_RAW_PATH = DATA_PATH/"raw"
DATA_RAW_EXTRACTED_PATH = DATA_PATH/"raw_data_extraction"

# Output is in bytes - helper from Pathlib Path https://stackoverflow.com/questions/2104080/how-can-i-check-file-size-in-python
def getStat(prev_value, cur_value):
    if isinstance(prev_value, int):
        return prev_value + cur_value.stat().st_size
    return prev_value.stat().st_size + cur_value.stat().st_size

# 1. The data from thwiki
THWIKI_FOLDER = Path("thwiki-20200601-extracted")
WIKI_FILES = list((DATA_RAW_EXTRACTED_PATH/THWIKI_FOLDER).glob("*.txt"))
list(map(print , WIKI_FILES[:5]))
print(f"thwiki-20200601-extracted Amounts to a total of {reduce(getStat, WIKI_FILES)/1e6:.2f} MB")

# 2. The classification data from jung and ninja
CLASSIFICATION_JUNG_NINJA_FOLDER = Path("classification_dataset")
CLASSIFICATION_FILES = list((DATA_RAW_EXTRACTED_PATH/CLASSIFICATION_JUNG_NINJA_FOLDER).glob("*.txt"))
list(map(print , CLASSIFICATION_FILES[:5]))
print(f"classification_dataset Amounts to a total of {reduce(getStat, CLASSIFICATION_FILES)/1e6:.2f} MB")

# 3. The Data from p'Moo Crawlers
ANOTHER_WEBSITE_MOO_FOLDER = Path("another_website")
ANOTHER_WEBSITE_FILES = list((DATA_RAW_EXTRACTED_PATH/ANOTHER_WEBSITE_MOO_FOLDER).glob("*.txt"))
list(map(print , ANOTHER_WEBSITE_FILES[:5]))
print(f"another_website Amounts to a total of {reduce(getStat, ANOTHER_WEBSITE_FILES)/1e6:.2f} MB")

# 4. Senior Project Files
SENIOR_PROJ_FOLDER = Path("data_lm")
SENIOR_PROJ_FILES = list((DATA_RAW_EXTRACTED_PATH/SENIOR_PROJ_FOLDER).glob("*.txt"))
list(map(print , SENIOR_PROJ_FILES[:5]))
print(f"Senior Project Amounts to a total of {reduce(getStat, SENIOR_PROJ_FILES)/1e6:.2f} MB")

# 5. Guru Crawler Files
GURU_CRAWLER_FOLDER = Path("social_listening")
GURU_CRAWLER_FILES = list((DATA_RAW_EXTRACTED_PATH/GURU_CRAWLER_FOLDER).glob("*.txt"))
list(map(print , GURU_CRAWLER_FILES[:5]))
print(f"GuruCrawler Amounts to a total of {reduce(getStat, GURU_CRAWLER_FILES)/1e6:.2f} MB")

ALL_FILES = WIKI_FILES + CLASSIFICATION_FILES + ANOTHER_WEBSITE_FILES + SENIOR_PROJ_FILES + GURU_CRAWLER_FILES
print(f"\nI have a total of {len(ALL_FILES)} files!")





print(f"Amounts to a total of {reduce(getStat, ALL_FILES)/1e6:.2f} MB")


class CustomSeniorProjectTokenizer(object):
    def __init__(self, TOK_PATH = Path('./senior_proj_itos'), BOS='xxbos', EOS='xxeos', FLD = 'xxfld', UNK='xxunk', PAD='xxpad',
                 TK_REP='xxrep', TK_WREP='xxwrep', TK_NUM='xxnum', TK_LAUGH='xxlaugh'
                ):
        from senior_project_util import ThaiTokenizer, pre_rules_th, post_rules_th
        from fastai.text.transform import BaseTokenizer, Tokenizer, Vocab
        from fastai.text.data import TokenizeProcessor, NumericalizeProcessor

        with open(TOK_PATH/"bert_itos_80k_cleaned.pkl", 'rb') as f:
            itos = pickle.load(f)
            
        self.vocab = Vocab(itos)
        self.tokenizer = Tokenizer(tok_func = ThaiTokenizer, lang = 'th', 
                                   pre_rules = pre_rules_th, post_rules=post_rules_th, n_cpus=1)
        
        self.cls_token_id = self.vocab.stoi[BOS]
        self.sep_token_id = self.vocab.stoi[EOS]
        
#         tokenizer_processor = TokenizeProcessor(tokenizer=tt, chunksize=300000, mark_fields=False)
#         numbericalize_processor = NumericalizeProcessor(vocab=vocab)
        
    def num_special_tokens_to_add(self, pair=False):
        return 2
    def tokenize(self, text):
        return self.tokenizer._process_all_1([text])[0]
#         return self.tokenizer.process_all([text])[0]
    
    def convert_tokens_to_ids(self, token_list):
        return self.vocab.numericalize(token_list)
    
    def build_inputs_with_special_tokens(self, token_list):
        # From https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_bert.py#L235
        return [self.cls_token_id] + token_list + [self.sep_token_id]
    
    
text = "ใครเคยมีแฟนที่กินอาหารไม่ถูกปากกันแล้วรู้สึกเสียความสุขไปอย่างนึงบ้างมั้ยครับ  ก่อนอื่นผมต้องบอกก่อนเลยว่าคนเราจะเลือกกินอาหารแบบไหนชอบแบบไหนเป็นเรื่องของความชอบส่วนตัวนะครับทุกคนมีสิทธิในการเลือกของที่ชอบและไม่ชอบอยู่แล้ว แต่ผมรู้สึกว่าตอนนี้ผมกำลังประสบปัญหาที่ดูเหมือนจะเล็กแต่กลายเป็นว่ามันค่อนข้างใหญ่ ผมคบกับแฟนมา6ปีแล้วครับ ผมเป็นคนชอบกินอาหารญี่ปุ่นและปลาดิบแต่แฟนผมไม่กินปลาดิบเลย ผมอยากกินบุฟเฟ่เนื้อแต่แฟนผมก็ไม่กินเนื้อ เราเลยไม่ได้เข้าทานร้านบุฟเฟ่เนื้อและบุฟเฟ่อาหารญี่ปุ่นกันเพราะรู้สึกลัวแฟนผมทานไม่คุ้ม และเรื่องใหญ่เลยคือผมเป็นคนชอบทานอาหารรสจัดและรสเผ็ดมาก แต่แฟนผมทานเผ็ดไม่ได้เลยเวลาเราไปกินส้มตำกันก็จะสั่ง ส้มตำไม่ใส่พริก ต้มแซ่บไม่ใส่พริก ลาบไม่ใส่พริก ร้านกับข้าวอื่นๆก็เช่นกันแฟนผมจะไม่ชอบกินผักไม่ค่อยสั่งกับข้าวที่เป็นผักแล้วผมชอบผักบุ้งทอดกรอบ เห็ดหอมสดทอดมาก แต่ก็ไม่ได้สั่งเพราะว่าเธอไม่กินถึงเค้าจะบอกให้สั่งเลยๆก็เถอะแต่ผมก็ยังเกรงใจเธออยู่ดีอ่ะครับ ผมรู้สึกกินอาหารไม่มีความสุขเลยชีวิตผมขาดรสเผ็ดไปเหมือนจะขาดใจเหมือนมันทำให้ขาดความสุขไปอย่างนึงเลยอ่ะครับ ยิ่งถ้าเราแต่งงานกันแล้วผมก็อาจจะต้องมีปัญหาเรื่องนี้มากขึ้น พอผมเห็นคู่ที่ชอบทานอาหารเหมือนๆกันเห็นเค้ากินอาหารกันอย่างมีความสุขแล้วผมรู้สึกอิจฉามากๆเลย มีใครเคยมีปัญหาแบบผมมั้ยครับแล้วจะแก้ปัญหานี้ยังไงดีครับ"
tokenizer = CustomSeniorProjectTokenizer()
print(tokenizer.num_special_tokens_to_add(pair=False))
print(tokenizer.__class__.__name__)
value = tokenizer.tokenize(text)
print(value)
value = tokenizer.convert_tokens_to_ids(value)
print(value)
value = tokenizer.build_inputs_with_special_tokens(value)
print(value)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)
class TextDatasetParallel(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """         
    def __init__(self, tokenizer: PreTrainedTokenizer, sample_path: [], block_size: int, overwrite_cache=False,
                num_processes=8, cached_directory = "/workdir/Code/bma_transformer_model/data/cached_data"):
        # assert os.path.isfile(file_path)
        # For Loop MultiFile
        self.examples = []
        self.sample_path = sample_path
#         print(f"THIS IS SAMPLE PATH {sample_path}")
        self.tokenizer = tokenizer
        
        # Set block size to be the blocksize-special tokens
        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        
        self.overwrite_cache = overwrite_cache
        self.cached_directory = cached_directory
        if not os.path.exists(cached_directory):
            os.makedirs(cached_directory)
        
        # Multiprocess for getting examples
        with Pool(processes=num_processes) as p:
            self.examples = list(tqdm.tqdm(p.imap(self.load_data_tokenized, self.sample_path), total=len(self.sample_path)))
        # Convert from 3d list to 2d 
        # self.examples from [[[3], [4]], [[5], [6]], [[7], [8]]] => [[3], [4], [5], [6], [7], [8]]
        self.examples = [each_batch for each_file in self.examples for each_batch in each_file]
        

    def load_data_tokenized(self, file_path):
#         print(f"I AM DOING {file_path}")
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            self.cached_directory, f"cached_lm_{tokenizer.__class__.__name__}_{str(self.block_size)}_{filename}",
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not self.overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    temp_examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                temp_examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
#                 print("I finished reading ", file_path)
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text[:100000]))
#                 print("I finished tokenizing ", file_path)
                for i in range(0, len(tokenized_text) - self.block_size + 1, self.block_size):  # Truncate in block of block_size
                    temp_examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + self.block_size])
                    )
#                     if i%20 == 0:
#                         print("I finished special tok ", file_path)
#                 print("I finished every tokenizing ", file_path)
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(temp_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )
        return temp_examples
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

    
tik = time.time()

dataset = TextDatasetParallel(tokenizer, 
                              sample_path=list(map(str, ALL_FILES)), 
#                               sample_path=list(map(str, GURU_CRAWLER_FILES)), 
                              block_size=512, 
                              cached_directory= "/workdir/cached_data",
                              overwrite_cache=False, # make sure this is false when you have cache!!
                              num_processes=40,
                             )
toc = time.time()
print(f"Process completed within {toc-tik}")