with open("data/raw/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("characters:", len(text))
print(text[:1000])

chars = sorted(set(text))
print("vocab size:", len(chars))
print(repr("".join(chars)))