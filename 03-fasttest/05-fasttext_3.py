import fasttext
import time

train_data_path = './train_fast1.txt'
dev_data_path = './dev_fast1.txt'
test_data_path = './test_fast1.txt'

model = fasttext.train_supervised(
    input=train_data_path,
    autotuneValidationFile=dev_data_path,
    autotuneDuration=6,
    wordNgrams=2,
    verbose=3
)

result = model.test(test_data_path)
print(result)
# 0.9162 -> 0.9154 变化不明显，可能是设备性能不足？或者用词为单位和字单位无区别？bin被我删了。
time1 = time.time()
model_save_path = './news_fasttext_{}.bin'.format(time1)
model.save_model(model_save_path)
