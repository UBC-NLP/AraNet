# AraNet: A Deep Learning Toolkit for Arabic Social Media
<img src="https://github.com/UBC-NLP/AraNet/blob/master/AraNet2.png" alt="drawing" width="50"/>

AraNet, a deep learning toolkit for a host of Arabic social media processing. AraNet predicts age, dialect, gender, emotion, irony, and sentiment from social media posts. It delivers either state-of-the-art or competitive performance on these tasks. It also has the advantage of using a unified, simple framework based on the recently-developed BERT model. AraNet has the potential to alleviate issues related to comparing across different Arabic social media NLP tasks, by providing one way to test new models against AraNet predictions (i.e., model-based comparisons). AraNet can be used to make important discoveries about the Arab world, a vast geographical region of strategic importance. It can enhance also enhance our understating of Arabic online communities, and the Arabic digital culture in general.

## How to install
 - Using pip
 ```shell
  pip install git+https://github.com/UBC-NLP/aranet
 ```
 - Clone and install
 ```shell
  git clone https://github.com/UBC-NLP/aranet
  cd aranet
  pip install .
```
## Download models
 - [Sentiment](https://drive.google.com/file/d/1W5171aT-1rYwK2iQyKGF0C2kT1UPjq11/view?usp=sharing)
 - [Dialect](https://drive.google.com/file/d/1e85Y1bhvPc9yjwKq-lSIewHIRgCDa4YS/view?usp=sharing)
 - [Emotion](https://drive.google.com/file/d/1D1nvw715Yp_yK6XYYPfUxnxygfPEEK2k/view?usp=sharing)
 - [Irony](https://drive.google.com/file/d/1FzWmCNISoWwGJbdNM65frDJi4QVs_-g1/view?usp=sharing)
 - Gender
 - Age
## How to use
You can easily add AraNet in your code
### Dialect 
```python
from aranet import aranet`
`#load AraNet dialect model
model_path = "./models/dialect_aranet/"
dialect_obj = aranet.AraNet(model_path)
```
``` python
tweet_text="انا هاخد ده لو سمحت"
dialect_obj.predict(text=tweet_text)
```
[('Egypt', 0.9993844)]
``` python
text_str="العشا اليوم كان عند الشيخ علي حمدي الحداد ، لمؤخذة بقى على الخيانة ، ايش مشاك غادي"
dialect_obj.predict(text=text_str)
```
[('Libya', 0.763)]
```python
text_str ="يعيشك برقا"
dialect_obj.predict(text=text_str)
```
[('Tunisia', 0.998887)]
### Sentiment 
```python
#load AraNet sentiment model
model_path = "./models/sentiment_aranet/"
senti_obj = aranet.AraNet(model_path)
```
```python
text_str ="ما اكره واحد قد هذا المنافق"
senti_obj.predict(text=text_str)
```
[('neg', 0.8975404)]
```python
text_str ="يعيشك برقا"
senti_obj.predict(text=text_str)
```
[('pos', 0.747435)]
### Emotion
```python
#load AraNet emotion model
model_path = "./models/emotion_aranet/"
emo_obj = aranet.AraNet(model_path) 
```
```python
text_str ="الله عليكي و انتي دائما مفرحانا"
emo_obj.predict(text=text_str)
```
[('happy', 0.89688617)]
```python
text_str ="لم اعرف المستحيل يوما"
emo_obj.predict(text=text_str)
```
[('trust', 0.27242294)]
## Reference:
Please cite our work: 
```
@inproceedings{mageed2020aranet,
  title={AraNet: A Deep Learning Toolkit for Arabic Social Media},
  author={Abdul-Mageed, Muhammad and Zhang, Chiyu and Hashemi, Azadeh and others},
  booktitle={Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing Tools, with a Shared Task on Offensive Language Detection},
  pages={16--23},
  year={2020}
}
```
