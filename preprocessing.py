import pandas as pd
import re
import numpy as np



df = pd.read_csv("./Dataset/datas/teapro.csv", encoding='utf-8')
df = df.dropna(subset=['rateContent'])

# 找出所有的非中文、非英文和非数字符号
additional_chars = set()
for t in list(df.rateContent):
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', str(t)))

# 一些需要保留的符号
extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")
# print(extra_chars)
additional_chars = additional_chars.difference(extra_chars)

def clean_data(x):
    try:
        x = x.strip()
    except:
        return ''
    x = re.sub('{IMG:.?.?.?}', '', x)
    x = re.sub('<!--IMG_\d+-->', '', x)
    x = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x)  # 过滤网址
    x = re.sub('<a[^>]*>', '', x).replace("</a>", "")  # 过滤a标签
    x = re.sub('<P[^>]*>', '', x).replace("</P>", "")  # 过滤P标签
    x = re.sub('<strong[^>]*>', ',', x).replace("</strong>", "")  # 过滤strong标签
    x = re.sub('<br>', ',', x)  # 过滤br标签
    x = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x).replace("()", "")  # 过滤www开头的网址
    x = re.sub('\s', '', x)   # 过滤不可见字符
    x = re.sub('Ⅴ', 'V', x)

    for wbad in additional_chars:
        x = x.replace(wbad, '')
    return x

# 清除噪音
df['rateContent'] = df['rateContent'].apply(clean_data)
df['rateContent'].replace('', np.nan, inplace=True)
cleaned_df = df.dropna(subset=['rateContent'])
cleaned_df = cleaned_df.rename(columns={'rateContent': 'text'})

cleaned_df.to_csv("./Dataset/datas/cleaned_data.csv", encoding='utf-8', index=False)

