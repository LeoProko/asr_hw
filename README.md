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

```bash
python3 test.py -r model_best.pth -c hw_asr/configs/deepspeech2/test.json
```
