import random

N = 10000

with open(r"c:\Users\chara\Desktop\URL-Phishing-Detecttion\dataset\train\train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

small_data = random.sample(lines, N)

with open("small_train.txt", "w", encoding="utf-8") as f:
    f.writelines(small_data)