import random

N = 3000

with open(r"c:\Users\chara\Desktop\URL-Phishing-Detecttion\dataset\test\test.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

small_data = random.sample(lines, N)

with open("small_test.txt", "w", encoding="utf-8") as f:
    f.writelines(small_data)