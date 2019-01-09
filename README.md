# NTUseg
基於 BiLSTM 及 ELMo，State-of-the-art 的開源中文斷詞系統。  
An open source state-of-the-art Chinese word segmentation system with BiLSTM and ELMo.


## Performance
![](https://i.imgur.com/H9w9EFm.png)
此專案提供圖中的 "character level ELMo" model 以及 "baseline model"，其中 "character level ELMo" model 是當前準確率最高。  
這兩個 model 都贏過目前常用的斷詞系統 Jieba 及 CKIP 許多。  
This repo provides the "character level ELMo" model and "baseline model" in the figure. Our "character level ELMo" model outperforms the previous state-of-the-art Chinese word segmentation (Ma et al. 2018), and also largely outerform "Jieba" and "CKIP", which are most popular toolkits in processing simplified/traditional Chinese text.


![](https://i.imgur.com/Iw0zffr.png)
當處理未見詞時，"character level ELMo" model 仍然保有不錯的正確率，相較於"baseline model"。  
When considering OOV accuracy, our "character level ELMo" model outperforms our "baseline model" about 5%.

## How to use?
### Download the code and models
  - `$ git clone https://github.com/voidism/ntuseg`
  - download [ELMoForManyLangs.zip](https://www.dropbox.com/s/eiya6ztmjopprsm/ELMoForManyLangs.zip?dl=0) and unzip it to the main directory
### Segment!
  ```
  from ntuseg import *
  seg = Seger(batch_size=64, device="cuda:0", embedding='elmo', elmo_use_cuda=True, mode="TW")
  seg.cut(["今天天氣真好啊!", "潮水退了就知道，誰沒穿褲子。"])
  # will return a list of lists of the segmented sentences.
  ```
#### Parameters:
  - batch_size: batch size for the word segmentation model, default: 64.
  - device: the GPU device to put you model, default: "cpu".
  - embedding: if you choose 'elmo', the model will be the "character level ELMo" model above; if you choose 'w2v', the model will be the "baseline model" above.
  - elmo_use_cuda: if you want your ELMo model be accelerated on GPU, use True, elsewise the ELMo model will be run on CPU. This param is no use when embedding="w2v".
  - mode: Seger will load different model according to the mode as listed below:
    - `TW`: trained on AS corpus, from Academia Sinica, Taiwan.
    - `HK`: trained on CityU corpus, from City University of Hong Kong.
    - `CN_MSR`: trained on MSR corpus, from Microsoft Research.
    - `CN_PKU` or `CN`: trained on PKU corpus, from Peking University.
  
