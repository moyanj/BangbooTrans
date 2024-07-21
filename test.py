import inference

model = inference.Inference("1721569732", model_base="Models")
out = model.eval("你说对吧")

print("".join(out))
