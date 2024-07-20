import inference

model = inference.Inference("1721495918", model_base="models")
out = model.eval_steam("你说对吧", max_length=100)

print("".join(out))
