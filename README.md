# AraNet
## How to install
 - Using pip\
  `pip install git+https://github.com/UBC-NLP/aranet`
 - Clone and install\
  `git clone https://github.com/UBC-NLP/aranet`\
  `cd aranet`\
  `pip install .`

## Download models
 - [Sentiment](https://drive.google.com/file/d/13_2OtzLDCPsVa3lLvPLmgzY-7mA9BeXM/view?usp=sharing)
 - [Dialect](https://drive.google.com/file/d/1JMQ10O5tlVKwMW9sbsUvWVpnWyuVAyz9/view?usp=sharing)
 - [Emotion](https://drive.google.com/file/d/191wozliqkD29jyDiTnrvKv28fUyS_st3/view?usp=sharing)
 - [Irony](https://drive.google.com/file/d/1TbaJ1_KRMPfGObdxBU3dCI6D-erWhIcQ/view?usp=sharing)
 - Gender
 - Age
## How to use
- load the model
```python
from aranet import aranet`
`#load AraNet dialect model
model_path = "./models/dialect_aranet/"
dialect_obj = aranet.AraNet(model_path)
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
