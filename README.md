# Pywordseg
基於 BiLSTM 及 ELMo 的 State-of-the-art 開源中文斷詞系統。  
An open source state-of-the-art Chinese word segmentation system with BiLSTM and ELMo.  

- arXiv paper link: https://arxiv.org/abs/1901.05816
- PyPI page: https://pypi.org/project/pywordseg/

## Performance
![](https://i.imgur.com/4WflkYS.png)
- 此專案提供圖中的 "character level ELMo" model 以及 "baseline" model，其中 "character level ELMo" model 是當前準確率最高。這兩個 model 都贏過目前常用的斷詞系統 [Jieba](https://github.com/fxsjy/jieba) (HMM-based) 及 [CKIP](http://ckipsvr.iis.sinica.edu.tw/) (rule-based) 許多。  
- This repo provides the "character level ELMo" model and "baseline" model in the figure. Our "character level ELMo" model outperforms the previous state-of-the-art Chinese word segmentation (Ma et al. 2018), and also largely outerform "[Jieba](https://github.com/fxsjy/jieba)" and "[CKIP](http://ckipsvr.iis.sinica.edu.tw/)", which are most popular toolkits in processing simplified/traditional Chinese text.

![](https://i.imgur.com/0vCz0ui.png)
- 當處理訓練時未見過的詞時，"character level ELMo" model 仍然保有不錯的正確率，相較於"baseline" model。  
- When considering OOV accuracy, our "character level ELMo" model outperforms our "baseline" model about 5%.

## Usage
### Requirements
- python >= 3.6 (do not use 3.5)
- pytorch 0.4
- overrides

### Install with Pip
  - `$ pip install pywordseg`
  - the module will automatically download the models while your first import within 1 minute.
  - if you use MacOS and encounter the [urllib.error.URLError](https://stackoverflow.com/questions/49183801/ssl-certificate-verify-failed-with-urllib) problem when downloading your models,  
  try `$ sudo /Applications/Python\ 3.6/Install\ Certificates.command` to bypass the certificate issue.

### Install manually
  - `$ git clone https://github.com/voidism/pywordseg`
  - download [ELMoForManyLangs.zip](https://www.dropbox.com/s/eiya6ztmjopprsm/ELMoForManyLangs.zip?dl=0) and unzip it to the `pywordseg/pywordseg` (the code of the ELMo model is from [HIT-SCIR](https://github.com/HIT-SCIR/ELMoForManyLangs), training by myself in character-level)
  - `$ pip install .` under the main directory

### Segment!
  ```python
  # import the module
  from pywordseg import *
  
  # declare the segmentor.
  seg = Wordseg(batch_size=64, device="cuda:0", embedding='elmo', elmo_use_cuda=True, mode="TW")
  
  # input is a list of raw sentences.
  seg.cut(["今天天氣真好啊!", "潮水退了就知道，誰沒穿褲子。"])
  
  # will return a list of lists of the segmented sentences.
  # [['今天', '天氣', '真', '好', '啊', '!'], ['潮水', '退', '了', '就', '知道', ',', '誰', '沒', '穿', '褲子', '。']]
  ```
#### Parameters:
  - **batch_size**: batch size for the word segmentation model, default: `64`.
  - **device**: the CPU/GPU device to run you model, default: `'cpu'`.
  - **embedding**: (default: `'w2v'`) 
    - `'elmo'`: the loaded model will be the "character level ELMo" model above, which runs slow.
    - `'w2v'`: the loaded model will be the "baseline model" above, which runs faster than `'elmo'`.
  - **elmo_use_cuda**: if you want your ELMo model be accelerated on GPU, use `True`, otherwise the ELMo model will be run on CPU. This param is no use when `embedding='w2v'`. default: `True`.
  - **mode**: `WordSeg` will load different model according to the mode as listed below: (default: `TW`)
    - `TW`: trained on AS corpus, from CKIP, Academia Sinica, Taiwan.
    - `HK`: trained on CityU corpus, from City University of Hong Kong, Hong Kong SAR.
    - `CN_MSR`: trained on MSR corpus, from Microsoft Research, China.
    - `CN_PKU` or `CN`: trained on PKU corpus, from Peking University, China.

### Include External Dictionary (Optional)
This feature was inspired by [CKIPTagger](https://github.com/ckiplab/ckiptagger#3-optional-create-dictionary).
  ```python
  # import the module
  from pywordseg import *
  
  # declare the segmentor.
  seg = Wordseg(batch_size=64, device="cuda:0", embedding='elmo', elmo_use_cuda=True, mode="TW")

  # create dictionary with their relative weights to prioritize.
  word_to_weight = {
    "來辦": 2.0,
    "你本人": 1.0,
    "或者是": 1.0,
    "有興趣": 1.0,
    "有興趣的": "2.0",
  }
  dictionary = construct_dictionary(word_to_weight)
  print(dictionary)
  # [(2, {'來辦': 2.0}), (3, {'你本人': 1.0, '或者是': 1.0, '有興趣': 1.0}), (4, {'有興趣的': 2.0})]

  # 1) segment without dictionary.
  seg.cut(["你本人或者是親屬有興趣的話都可以來辦理"])
  # [['你', '本人', '或者', '是', '親屬', '有', '興趣', '的話', '都', '可以', '來', '辦理']]

  # 2) segment with dictionary to merge words (only merge words that will not break existing words).
  seg.cut(["你本人或者是親屬有興趣的話都可以來辦理"], merge_dict=dictionary)
  # [['你本人', '或者是', '親屬', '有興趣', '的話', '都', '可以', '來', '辦理']]
  # merged: '你', '本人' --> '你本人'
  # merged: '或者', '是' --> '或者是'
  # merged: '有', '興趣' --> '有興趣'
  # not merged: '來', '辦理' -x-> '來辦', '理' because it breaks existing words

  # 3) segment with dictionary that force words to be segmented (ignore existing words).
  seg.cut(["你本人或者是親屬有興趣的話都可以來辦理"], force_dict=dictionary)
  # [['你本人', '或者是', '親屬', '有興趣的', '話', '都', '可以', '來辦', '理']]
  # merged: '你', '本人' --> '你本人'
  # merged: '或者', '是' --> '或者是'
  # change: '有興趣', '的話' --> '有興趣的', '話'
  # change: '來', '辦理' --> '來辦', '理'
  ```

## TODO
- 目前只支援繁體中文(即使選擇CN mode，文字也要轉換成繁體才能運作，目前訓練資料都是經過 [OpenCC](https://github.com/BYVoid/OpenCC) 轉換的)，日後會加入簡體中文。

## Citation
If you use the code in your paper, then please cite it as:

    @article{Chuang2019,
      archivePrefix = {arXiv},
      arxivId       = {1901.05816},
      author        = {Chuang, Yung-Sung},
      eprint        = {1901.05816},
      title         = {Robust Chinese Word Segmentation with Contextualized Word Representations},
      url           = {http://arxiv.org/abs/1901.05816},
      year          = {2019}
    }
