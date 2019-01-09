# NTUseg
基於 BiLSTM 及 ELMo，當前 State-of-the-art 的開源中文斷詞系統。  
An open source state-of-the-art Chinese word segmentation system with BiLSTM and ELMo.


## Performance
![](https://i.imgur.com/H9w9EFm.png)
此專案提供圖中的 "character level ELMo" model 以及 "baseline model"，其中 "character level ELMo" model 是當前準確率最高。  
這兩個 model 都贏過目前常用的斷詞系統 Jieba 及 CKIP 許多。  
This repo provides the "character level ELMo" model and "baseline model" in the figure. Our "character level ELMo" model outperforms the previous state-of-the-art Chinese word segmentation (Ma et al. 2018), and also largely outerform "Jieba" and "CKIP", which are most popular toolkits in processing simplified/traditional Chinese text.


![](https://i.imgur.com/Iw0zffr.png)
當處理未見詞時，"character level ELMo" model 仍然保有不錯的正確率，相較於"baseline model"。  
When considering OOV accuracy, our "character level ELMo" model outperforms our "baseline model" about 5%.

