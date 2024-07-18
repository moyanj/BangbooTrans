import inference
model = inference.Inference('1721321831')
out = model.eval_steam('谢谢你！法厄同',max_length=100)

for char in out:
    print(char,end='')