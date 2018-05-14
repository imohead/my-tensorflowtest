from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
#カウント処理ライブラリ
from collections import Counter
import os

import numpy as np
np.set_printoptions(precision=10)
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from janome.tokenizer import Tokenizer

#メール件名
data = pd.read_csv("email_train_tag.csv")
print(data.head())
print(data['タグ'].value_counts())

# トレーニングとテストを分割
train_size = int(len(data) * .8)

#トレーニング用
train_posts = data['件名'][:train_size]
train_tags = data['タグ'][:train_size]
# テスト用
test_posts = data['件名'][train_size:]
test_tags = data['タグ'][train_size:]

max_words = 1000

#分かち書きにする
t = Tokenizer()
wakati = []
for subject in data['件名']:
    word = t.tokenize(subject, wakati=True)
    wakati.append(word)
    while wakati.count('') > 0:
        wakati.remove('')

#print(wakati)

# 出現数の多い単語をカウントして、多い順に並び替え
word_freq = Counter(itertools.chain(* wakati))
#辞書作成
dic = []
for word_uniq in word_freq.most_common():
    dic.append(word_uniq[0])

dic_inv = {}
for i, word_uniq in enumerate(dic, start=1):
    dic_inv.update({word_uniq: i})
print(dic_inv)

# トレーニング用を分かち書き
train_wakati = []
for wd in train_posts:
    word = t.tokenize(wd, wakati=True)
    train_wakati.append(word)
    while train_wakati.count('') > 0:
        train_wakati.remove('')

# テスト用を分かち書き
test_wakati = []
for wd in test_posts:
    word = t.tokenize(wd, wakati=True)
    test_wakati.append(word)
    while test_wakati.count('') > 0:
        test_wakati.remove('')


#変換した単語データ
x_train = [[dic_inv[word] for word in waka] for waka in train_wakati]
x_test = [[dic_inv[word] for word in waka] for waka in test_wakati]

#単語データの配列のサイズを揃える
x_train = sequence.pad_sequences(
    x_train, maxlen=max_words, dtype='int32', padding='post', truncating='post')
x_test = sequence.pad_sequences(
    x_test, maxlen=max_words, dtype='int32', padding='post', truncating='post')
#print(x_train)

#タグデータ
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

#タグデータをone-hot形式へ変換 1か0に変換
num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
#print(y_train)

# バッチサイズとトレーニング回数を指定
batch_size = 32
epochs = 2

# モデルを作成
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# トレーニング開始
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

# モデルの精度を確認
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# テストの結果を確認 1,0=Bad, 0,1=Good
text_labels = encoder.classes_

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(prediction)
    print(test_posts.iloc[i][:50], "...")
    print('Actual label:' + test_tags.iloc[i])
    print("Predicted label: " + predicted_label + "\n")

###########予測###########
#テキストを辞書のIDに変換してベクター化
def text_to_vector(text):
    t = Tokenizer()
    wordwakati = ""
    wordwakati = t.tokenize(text, wakati=True)
    word_vector = []
    return_vector = []  # 配列の中に配列化のため
    for word in wordwakati:
        for txt in dic_inv:
            if(word == txt):
                word_vector.append(dic_inv[word])
    #return word_vector
    return_vector.append(word_vector)
    return return_vector

kenmei = 'このメールはテストの件名です'
print(t.tokenize(kenmei, wakati=True))
text = text_to_vector(kenmei)  # ベクトル化
print(text)
#配列のサイズをそろえる
text = sequence.pad_sequences(text, maxlen=max_words, dtype='int32', padding='post', truncating='post')
print(text)
prediction = model.predict(np.array([text][0]))
predicted_label = text_labels[np.argmax(prediction)]
print(prediction)
print("Predicted label: " + predicted_label + "\n")
