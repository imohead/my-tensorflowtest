
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
import numpy.random as nr
import sys
import h5py
import math

from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.optimizers import RMSprop
from keras.utils import np_utils
# Janomeのロード
from janome.tokenizer import Tokenizer
# Word2Vecライブラリのロード
from gensim.models import word2vec


##### ファイル読込み、内部表現化 #####
f = open('hashire_merosu.txt', encoding='sjis')
text_sjis = f.read()
f.close()
#text = text_sjis.decode('sjis')
text = text_sjis

# ファイル整形
import re
# ヘッダ部分の除去
text = re.split(u'\-{5,}', text)[2]
# フッタ部分の除去
text = re.split(u'底本：', text)[0]
# | の除去
text = text.replace(u'|', u'')
# ルビの削除
text = re.sub(u'《.+?》', u'', text)
# 入力注の削除
text = re.sub(u'［＃.+?］', u'', text)
# 空行の削除
text = re.sub(u'\n\n', '\n', text)
text = re.sub(u'\r', '', text)

# 整形結果確認

# 頭の100文字の表示
#print(text[:100])

#print("\n\r")
# 後ろの100文字の表示
#print(text[-100:])

# 分かち書きにする
# Tokenneizerインスタンスの生成
t = Tokenizer()

# テキストを引数として、形態素解析の結果、名詞・動詞原型のみを配列で抽出する関数を定義


def extract_words(text):
    tokens = t.tokenize(text)
    return [token.base_form for token in tokens
            if token.part_of_speech.split(',')[0] in[u'名詞', u'動詞']]


#  関数テスト
ret = extract_words(u'メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。')
for word in ret:
    print(word)

# 全体のテキストを句点(u'。')で区切った配列にする。
sentences = text.split(u'。')
# それぞれの文章を単語リストに変換(処理に数分かかります)
words = [extract_words(sentence) for sentence in sentences]

# 結果の一部を確認
for word in words[0]:
    print(word)


##### 辞書データの作成#####
data1 = []
for wds in words:
    for wd in wds:
        data1.append(wd)
print(data1)
mat = np.array(data1)
words = sorted(list(set(mat)))
cnt = np.zeros(len(words))

print('total words:', len(words))
word_indices = dict((w, i) for i, w in enumerate(words))  # 単語をキーにインデックス検索
indices_word = dict((i, w) for i, w in enumerate(words))  # インデックスをキーに単語を検索

# 単語の出現数をカウント
for j in range(0, len(mat)):
  cnt[word_indices[mat[j]]] += 1

# 出現頻度の少ない単語を「UNK」で置き換え
words_unk = []                           # 未知語一覧

for k in range(0, len(words)):
  if cnt[k] <= 3:
    words_unk.append(words[k])
    words[k] = 'UNK'

print('低頻度語数:', len(words_unk))    # words_unkはunkに変換された単語のリスト

words = sorted(list(set(words)))
print('total words:', len(words))
word_indices = dict((w, i) for i, w in enumerate(words))  # 単語をキーにインデックス検索
indices_word = dict((i, w) for i, w in enumerate(words))  # インデックスをキーに単語を検索

##### 訓練データ作成 #####
maxlen = 10                   # 前後の語数

mat_urtext = np.zeros((len(mat), 1), dtype=int)
for i in range(0, len(mat)):
  # 出現頻度の低い単語のインデックスをunkのそれに置き換え
  if mat[i] in word_indices:
    mat_urtext[i, 0] = word_indices[mat[i]]
  else:
    mat_urtext[i, 0] = word_indices['UNK']

print(mat_urtext.shape)

len_seq = len(mat_urtext) - maxlen
data = []
target = []
for i in range(maxlen, len_seq):
  data.append(mat_urtext[i])
  target.extend(mat_urtext[i-maxlen:i])
  target.extend(mat_urtext[i+1:i+1+maxlen])

x_train = np.array(data).reshape(len(data), 1)
t_train = np.array(target).reshape(len(data), maxlen*2)

z = list(zip(x_train, t_train))
nr.seed(12345)
nr.shuffle(z)                 # シャッフル
x_train, t_train = zip(*z)

x_train = np.array(x_train).reshape(len(data), 1)
t_train = np.array(t_train).reshape(len(data), maxlen*2)

print(x_train.shape, t_train.shape)

##### ニューラルネットワーク構築 #####
class Prediction:
  def __init__(self, input_dim, output_dim):
    self.input_dim = input_dim
    self.output_dim = output_dim

  def create_model(self):
    model = Sequential()
    model.add(Embedding(self.input_dim, self.output_dim,
                        input_length=1, embeddings_initializer=uniform(seed=20170719)))
    model.add(Flatten())
    model.add(Dense(self.input_dim, use_bias=False,
                    kernel_initializer=glorot_uniform(seed=20170719)))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="RMSprop", metrics=['categorical_accuracy'])
    print('#2')
    return model

  # 学習
  def train(self, x_train, t_train, batch_size, epochs, maxlen, emb_param):
    early_stopping = EarlyStopping(
        monitor='categorical_accuracy', patience=1, verbose=1)
    print('#1', t_train.shape)
    model = self.create_model()
    #model.load_weights(emb_param)    # 埋め込みパラメーターセット。ファイルをロードして学習を再開したいときに有効にする
    print('#3')
    model.fit(x_train, t_train, batch_size=batch_size, epochs=epochs, verbose=1,
              shuffle=True, callbacks=[early_stopping], validation_split=0.0)
    return model


##### メイン処理 #####
vec_dim = 100
epochs = 10
batch_size = 200
input_dim = len(words)
output_dim = vec_dim

emb_param = 'param_skip_gram_2_1.hdf5'    # 学習済みパラメーターファイル名
prediction = Prediction(input_dim, output_dim)
row = t_train.shape[0]

t_one_hot = np.zeros((row, input_dim), dtype='int8')    # ラベルデータをN-hot化
for i in range(0, row):
  for j in range(0, maxlen*2):
    t_one_hot[i, t_train[i, j]] = 1

x_train = x_train.reshape(row, 1)
model = prediction.train(x_train, t_one_hot, batch_size,
                         epochs, maxlen, emb_param)

model.save_weights(emb_param)           # 学習済みパラメーターセーブ

##### 評価 #####
param_list = model.get_weights()
param = param_list[0]
word0 = '男'
word1 = 'セリヌンティウス'
word2 = '走る'
vec0 = param[word_indices[word0], :]
vec1 = param[word_indices[word1], :]
vec2 = param[word_indices[word2], :]

vec = vec0 - vec1 + vec2
vec_norm = math.sqrt(np.dot(vec, vec))

w_list = [word_indices[word0], word_indices[word1], word_indices[word2]]
dist = -1.0
m = 0
for j in range(0, 5):
  dist = -1.0
  m = 0
  for i in range(0, len(words)):
    if i not in w_list:
      dist0 = np.dot(vec, param[i, :])
      dist0 = dist0 / vec_norm / math.sqrt(np.dot(param[i, :], param[i, :]))
      if dist < dist0:
        dist = dist0
        m = i
  print('第' + str(j+1) + '候補:')
  print('コサイン類似度=', dist, ' ', m, ' ', indices_word[m])
  w_list.append(m)
