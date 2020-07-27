from pathlib import Path
import re
import pythainlp
try:
    import cPickle as pickle
except:
    import pickle
import os
from pythainlp.ulmfit.preprocess import (
    lowercase_all,
    replace_rep_after,
    rm_brackets,
    rm_useless_newlines,
    ungroup_emoji,
)
from fastai.text.transform import fix_html, rm_useless_spaces, spec_add_spaces, replace_all_caps, replace_rep
from fastai.text.transform import BaseTokenizer, Tokenizer
from fastai.text.data import TokenizeProcessor, NumericalizeProcessor

from pythainlp.util import normalize as normalize_char_order
from pythainlp.tokenize import Tokenizer as PyThaiTokenizer

github_path = Path('../thai-lm')

TK_LAUGH = 'xxlaugh'

def replace_laugh(t:str) -> str:
    """Replaces numbers with TK_NUM"""
    def _replace_laugh(m) -> str:
        return f' {TK_LAUGH} '
    re_num = re.compile(r'(5)(\1{2,})(\+)*')
    return re_num.sub(_replace_laugh, t)

TK_NUM = 'xxnum'

def replace_number(t) -> str:
    """Replaces numbers with TK_NUM"""
    def _replace_number(m) -> str:
        return f' {TK_NUM} '
    re_num = re.compile(r'(([0-9|๐๑๒๓๔๕๖๗๘๙]+[,.-]?([0-9|๐๑๒๓๔๕๖๗๘๙x]+[,.-]?)+))|(?<!xxfld )([0-9|๐๑๒๓๔๕๖๗๘๙]+[,.-]?)+')
    return re_num.sub(_replace_number, t)

TK_REP = 'xxrep'

def new_replace_rep_after(t):
    "Replace repetitions at the character level in `t` after the repetition"

    def _replace_rep(m):
        c, cc = m.groups()
        return f"{c} {TK_REP} {min(len(cc)+1,5)} "

    re_rep = re.compile(r"(\S)(\1{2,})")
    return re_rep.sub(_replace_rep, t)

TK_WREP = 'xxwrep'

def replace_wrep(t):
    """Replace words (characters 2 or more) that have more than 3 repetitions. 
    Beware !!! run replace_rep_after first !!! กกกกกก can be กก grouped 3 times!!!
    (there is a way to see unique characters but too complicated)"""
    def _replace_wrep(m):
        c, cc = m.groups()
        return f" {c} {TK_WREP} {min(cc.count(c),5)} "
    
    re_wrep = re.compile(r"(\S{2,}?)(\1{2,})")
    return re_wrep.sub(_replace_wrep, t)

def correct_wrong(x):
    "Replace all wrong words (normalize) with replace_dict"
    with open('replace_dict.pkl', 'rb') as f:
        _replace_dict = pickle.load(f)
        res = []
        for t in x:
            if t in _replace_dict: res.append(_replace_dict[t])
            else: res.append(t)
        return res
    
class ThaiTokenizer(BaseTokenizer):
    def __init__(self, lang='th'):
        self.lang = lang
        self.pyengine = PyThaiTokenizer(os.path.join(github_path, 'words_modified.txt'))
    
    def tokenizer(self, t):
        return self.pyengine.word_tokenize(t)
    
# Preprocessing rules for Thai text
pre_rules_th = [fix_html, replace_laugh, replace_number, new_replace_rep_after, replace_wrep, 
                normalize_char_order, spec_add_spaces, rm_useless_spaces, rm_useless_newlines, rm_brackets]
post_rules_th = [ungroup_emoji, lowercase_all, correct_wrong]
