import inference
model = inference.Inference('BangbooTrans-104M-1831',model_base='Models')
out = model.eval_steam('你说对吧',max_length=100)

print(''.join(out))