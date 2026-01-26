import fasttext
import time

train_data_path = './train_fast.txt'
dev_data_path = './dev_fast.txt'
test_data_path = './test_fast.txt'

model = fasttext.train_supervised(
    input=train_data_path,
    autotuneValidationFile=dev_data_path,
    autotuneDuration=6,
    wordNgrams=2,
    verbose=3
)

result = model.test(test_data_path)
print(result)
# 0.9162
time1 = time.time()
model_save_path = './news_fasttext_{}.bin'.format(time1)
model.save_model(model_save_path)
