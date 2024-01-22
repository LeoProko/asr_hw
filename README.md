# ASR

В этом проекте реализована модель DeepSpeech2 для распознования текст из аудио

# Пререквизиты

Необходимо настроить виртуальное окружения для python3.10 из файла
requirements.txt

# Train

```bash
python3 train.py -c hw_asr/configs/deepspeech2/train.json
```

# Test

Скачать чекпоинт можно по ссылке

https://drive.google.com/file/d/1yOyxEZHTUQ3nl_rFLy7eMLwdJMAq5Cla/view?usp=sharing

```bash
python3 test.py -r deepspeech2.pth -c hw_asr/configs/deepspeech2/test.json
```
