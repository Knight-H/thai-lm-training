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
# from concurrent.futures import ProcessPoolExecutor as Pool
from functools import reduce


# Suggested Parsing from https://pytorch.org/docs/stable/distributed.html#launch-utility
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
print(f"THIS IS ARGS {args}")

# torch.cuda.set_device(arg.local_rank)
# In[5]:


DATA_PATH = Path("/workdir/Code/bma_transformer_model/data")

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


# # Trying out BERT per Notebook 
# 
# From __HuggingFace Notebooks__ https://huggingface.co/transformers/notebooks.html: 
# 
# How to train a language model	Highlight all the steps to effectively train Transformer model on custom data
# - Colab (ipynb) version : https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
# - MD version: https://github.com/huggingface/blog/blob/master/how-to-train.md
# 
# Pretrain Longformer	How to build a "long" version of existing pretrained models	Iz Beltagy  
# https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb

# In[6]:

from transformers import BertForMaskedLM, BertConfig

configuration = BertConfig(
    vocab_size=80000,
#     max_position_embeddings=512, # 512 + 2 more special tokens
#     num_attention_heads=12,
#     num_hidden_layers=12,
#     type_vocab_size=1,
)
# configuration.vocab_size = 20000

model = BertForMaskedLM(config=configuration)
# model = RobertaForMaskedLM.from_pretrained('./Roberta/checkpoint-200000')

# Accessing the model configuration
# model.config

# # Initializing Tokenizer

# ## Rewrite Tokenizer of bert_itos_80k with special tokens in front

# In[9]:


from senior_project_util import ThaiTokenizer, pre_rules_th, post_rules_th
from fastai.text.transform import BaseTokenizer, Tokenizer, Vocab
from fastai.text.data import TokenizeProcessor, NumericalizeProcessor

TOK_PATH = Path('./senior_proj_itos')

max_vocab = 80000

BOS,EOS,FLD,UNK,PAD = 'xxbos','xxeos','xxfld','xxunk','xxpad'
TK_REP,TK_WREP, TK_NUM, TK_LAUGH = 'xxrep','xxwrep', 'xxnum', 'xxlaugh'
text_spec_tok = [UNK,PAD,BOS,EOS,FLD,TK_REP,TK_WREP, TK_NUM, TK_LAUGH]


# In[10]:


import fastai
print(f"Running on Fastai version: {fastai.__version__}")


# In[15]:


with open(TOK_PATH/"bert_itos_80k_cleaned.pkl", 'rb') as f:
    itos = pickle.load(f)
# len(itos)


# In[16]:


vocab = Vocab(itos)


# In[18]:


# tt = Tokenizer(tok_func = ThaiTokenizer, lang = 'th', pre_rules = pre_rules_th, post_rules=post_rules_th, n_cpus=1)
# test_sample = tt._process_all_1([text[:100]])
# print(test_sample)
# test_sample = [vocab.numericalize(seq) for seq in test_sample]
# print(test_sample)




# In[21]:


class CustomSeniorProjectTokenizer(object):
    def __init__(self, TOK_PATH = Path('./senior_proj_itos'), BOS='xxbos', EOS='xxeos', FLD = 'xxfld', UNK='xxunk', PAD='xxpad',
                 TK_REP='xxrep', TK_WREP='xxwrep', TK_NUM='xxnum', TK_LAUGH='xxlaugh', n_cpus=1,
                ):
        from senior_project_util import ThaiTokenizer, pre_rules_th, post_rules_th
        from fastai.text.transform import BaseTokenizer, Tokenizer, Vocab
        from fastai.text.data import TokenizeProcessor, NumericalizeProcessor

        with open(TOK_PATH/"bert_itos_80k_cleaned.pkl", 'rb') as f:
            itos = pickle.load(f)
            
        self.vocab = Vocab(itos)
        self.tokenizer = Tokenizer(tok_func = ThaiTokenizer, lang = 'th', 
                                   pre_rules = pre_rules_th, post_rules=post_rules_th, n_cpus=n_cpus)
        
        self.cls_token_id = self.vocab.stoi[BOS]
        self.sep_token_id = self.vocab.stoi[EOS]
        self.pad_token_id = self.vocab.stoi[PAD]
        
        self.mask_token = FLD  #SINCE THIS ONE IS NOT USED, and INSIDE SPECIAL TOKEN....
        self._pad_token = PAD
        
#         tokenizer_processor = TokenizeProcessor(tokenizer=tt, chunksize=300000, mark_fields=False)
#         numbericalize_processor = NumericalizeProcessor(vocab=vocab)
        
    def num_special_tokens_to_add(self, pair=False):
        return 2
    def tokenize(self, text):
        return self.tokenizer._process_all_1([text])[0]
#         return self.tokenizer.process_all([text])[0]
    
    def convert_tokens_to_ids(self, token_list):
        #From https://huggingface.co/transformers/_modules/transformers/tokenization_utils_fast.html#PreTrainedTokenizerFast.convert_tokens_to_ids
        if token_list is None:
            return None

        if isinstance(token_list, str):
            return self.vocab.numericalize([token_list])[0]
        
        return self.vocab.numericalize(token_list)
    
    def build_inputs_with_special_tokens(self, token_list):
        # From https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_bert.py#L235
        return [self.cls_token_id] + token_list + [self.sep_token_id]
    
    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1 = None, already_has_special_tokens = False
    ):
        # From https://huggingface.co/transformers/_modules/transformers/tokenization_utils.html#PreTrainedTokenizer.get_special_tokens_mask
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))
    
    def __len__(self):
        #https://huggingface.co/transformers/_modules/transformers/tokenization_utils_fast.html#PreTrainedTokenizerFast.__len__
        return len(self.vocab.itos)


# In[22]:


# text = "ใครเคยมีแฟนที่กินอาหารไม่ถูกปากกันแล้วรู้สึกเสียความสุขไปอย่างนึงบ้างมั้ยครับ  ก่อนอื่นผมต้องบอกก่อนเลยว่าคนเราจะเลือกกินอาหารแบบไหนชอบแบบไหนเป็นเรื่องของความชอบส่วนตัวนะครับทุกคนมีสิทธิในการเลือกของที่ชอบและไม่ชอบอยู่แล้ว แต่ผมรู้สึกว่าตอนนี้ผมกำลังประสบปัญหาที่ดูเหมือนจะเล็กแต่กลายเป็นว่ามันค่อนข้างใหญ่ ผมคบกับแฟนมา6ปีแล้วครับ ผมเป็นคนชอบกินอาหารญี่ปุ่นและปลาดิบแต่แฟนผมไม่กินปลาดิบเลย ผมอยากกินบุฟเฟ่เนื้อแต่แฟนผมก็ไม่กินเนื้อ เราเลยไม่ได้เข้าทานร้านบุฟเฟ่เนื้อและบุฟเฟ่อาหารญี่ปุ่นกันเพราะรู้สึกลัวแฟนผมทานไม่คุ้ม และเรื่องใหญ่เลยคือผมเป็นคนชอบทานอาหารรสจัดและรสเผ็ดมาก แต่แฟนผมทานเผ็ดไม่ได้เลยเวลาเราไปกินส้มตำกันก็จะสั่ง ส้มตำไม่ใส่พริก ต้มแซ่บไม่ใส่พริก ลาบไม่ใส่พริก ร้านกับข้าวอื่นๆก็เช่นกันแฟนผมจะไม่ชอบกินผักไม่ค่อยสั่งกับข้าวที่เป็นผักแล้วผมชอบผักบุ้งทอดกรอบ เห็ดหอมสดทอดมาก แต่ก็ไม่ได้สั่งเพราะว่าเธอไม่กินถึงเค้าจะบอกให้สั่งเลยๆก็เถอะแต่ผมก็ยังเกรงใจเธออยู่ดีอ่ะครับ ผมรู้สึกกินอาหารไม่มีความสุขเลยชีวิตผมขาดรสเผ็ดไปเหมือนจะขาดใจเหมือนมันทำให้ขาดความสุขไปอย่างนึงเลยอ่ะครับ ยิ่งถ้าเราแต่งงานกันแล้วผมก็อาจจะต้องมีปัญหาเรื่องนี้มากขึ้น พอผมเห็นคู่ที่ชอบทานอาหารเหมือนๆกันเห็นเค้ากินอาหารกันอย่างมีความสุขแล้วผมรู้สึกอิจฉามากๆเลย มีใครเคยมีปัญหาแบบผมมั้ยครับแล้วจะแก้ปัญหานี้ยังไงดีครับ"
# tokenizer = CustomSeniorProjectTokenizer()
# print(tokenizer.num_special_tokens_to_add(pair=False))
# print(tokenizer.__class__.__name__)
# value = tokenizer.tokenize(text)
# print(value)
# value = tokenizer.convert_tokens_to_ids(value)
# print(value)
# value = tokenizer.build_inputs_with_special_tokens(value)
# print(value)




# Constructing tokenizer wrapper based on [@theblackcat102 #259](https://github.com/huggingface/tokenizers/issues/259#issuecomment-625905930)

# # Building our dataset
# 
# Build it with `from torch.utils.data.dataset import Dataset` just like [TextDataset](https://github.com/huggingface/transformers/blob/448c467256332e4be8c122a159b482c1ef039b98/src/transformers/data/datasets/language_modeling.py) and [LineByLineTextDataset](https://github.com/huggingface/transformers/blob/448c467256332e4be8c122a159b482c1ef039b98/src/transformers/data/datasets/language_modeling.py#L78)
# 
# Note: Training with multiple files is currently not supported [issue/3445](https://github.com/huggingface/transformers/issues/3445)
# 
# padding documentation [link](https://github.com/huggingface/tokenizers/blob/master/bindings/python/tokenizers/implementations/base_tokenizer.py#L52)
# 
# Potential Improvements
# - การทำให้ Dataset นั้น dynamically tokenize + dynamically open file : ตอนนี้เวลาทำ Dataset จาก torch.utils.data.dataset จะทำการ tokenize เลยตอนอยู่ใน constructor  , กำลังคิดว่าถ้าเกิดว่า Data ใหญ่มากๆ อาจจะไม่เหมาะสมกับการทำแบบนี้  เพราะว่า Ram จะต้องมีขนาดเท่าๆกับ data ที่เราใส่เข้าไป  ซึ่งเป็นไปได้ยากหาก Data มีขนาดใหญ่มากๆ   ผมได้ทำการ Search ดูแล้วก็พบว่าจาก Discussion Forum ของ Pytorch: https://discuss.pytorch.org/t/how-to-use-a-huge-line-corpus-text-with-dataset-dataloader/30872 
# Option1: ใช้ pd.Dataframe ในการเปิด File แบบ small chunks of data https://discuss.pytorch.org/t/data-processing-as-a-batch-way/14154/4?u=ptrblck
# Option2: ใช้ byte Offsets จากไฟล์ใหญ่ๆเพื่อที่จะ lookup .seek(): https://github.com/pytorch/text/issues/130#issuecomment-510412877
# More Examples: https://github.com/pytorch/text/blob/master/torchtext/datasets/unsupervised_learning.py , https://github.com/pytorch/text/blob/a5880a3da7928dd7dd529507eec943a307204de7/examples/text_classification/iterable_train.py#L169-L214

# In[23]:


logger = logging.getLogger(__name__)
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
#         with Pool(max_workers=num_processes) as p:
#             self.examples = list(tqdm.tqdm(p.map(self.load_data_tokenized, self.sample_path), total=len(self.sample_path)))
#         for path in tqdm.tqdm(self.sample_path):
#             self.examples.append(self.load_data_tokenized(path))
        
        
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
#                 print(f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start)
            else:
                temp_examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
#                 print("I finished reading ", file_path)
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
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

tokenizer = CustomSeniorProjectTokenizer()
dataset = TextDatasetParallel(tokenizer, 
                              sample_path=list(map(str, ALL_FILES)), 
#                               sample_path=list(map(str, GURU_CRAWLER_FILES)), 
                              block_size=512, 
                              cached_directory= "/workdir/Code/bma_transformer_model/data/cached_data_senior",
                              overwrite_cache=False, # make sure this is false when you have cache!!
                              num_processes=8,
                             )

# In[45]:


from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)



# # Transfomers Trainer [link](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L133)
# 
# ```python
# class Trainer:
#     """
#     Trainer is a simple but feature-complete training and eval loop for PyTorch,
#     optimized for Transformers.
#     Args:
#         prediction_loss_only:
#             (Optional) in evaluation and prediction, only return the loss
#     """
#     def __init__(
#         self,
#         model: PreTrainedModel,
#         args: TrainingArguments,
#         data_collator: Optional[DataCollator] = None,
#         train_dataset: Optional[Dataset] = None,
#         eval_dataset: Optional[Dataset] = None,
#         compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
#         prediction_loss_only=False,
#         tb_writer: Optional["SummaryWriter"] = None,
#         optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
# ```
# 
# [TrainingArguments](https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py#L33) is referenced here. 

# In[46]:


from transformers import Trainer, TrainingArguments

# For Distributed Run (see https://pytorch.org/docs/stable/distributed.html#launch-utility):
#    python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="192.168.11.36" --master_port=1234 runGuru_pretrain_originalBert.py
#    OR
#    python -m torch.distributed.launch --nproc_per_node=2 runGuru_pretrain_originalBert.py

training_args = TrainingArguments(
    output_dir="./OriginalBert",
    overwrite_output_dir=False,  #"Use this to continue training if output_dir points to a checkpoint directory."
    
    fp16=True,
    fp16_opt_level='O3',
    
    local_rank=args.local_rank, # FOR DISTRIBUTED TRAINING!! EXPERIMENTAL
    
    
    do_train=True, #Whether to run training.
#     do_eval=True, #Whether to run eval on the dev set.
#     do_predict=True, # Whether to run predictions on the test set.
    
    num_train_epochs=200, # Total number of training epochs to perform.
    
    
    per_device_train_batch_size=6, # Batch size per GPU/TPU core/CPU for training.
#     per_device_eval_batch_size=256, # Batch size per GPU/TPU core/CPU for evaluation.
    
    learning_rate=5e-5,  #The initial learning rate for Adam.
    weight_decay=0.0,
    max_grad_norm=1.0,
    adam_epsilon=1e-8, #Epsilon for Adam optimizer.
    
    #Logging
#     logging_dir='', default_logdir -> return os.path.join("runs", current_time + "_" + socket.gethostname())
    logging_first_step= True,
    logging_steps = 500,
    
    save_steps=10_000,  #Save checkpoint every X updates steps.
    save_total_limit=2, #"Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints
    
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
#     eval_dataset=val_dataset
)


# Note : This is why the GPU 1 is having all the load, and this is how it can be mitigated, and how to migrate to distributed parallel https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255


trainer.train()
# In[ ]:


trainer.save_model("./OriginalBert_Final2")
