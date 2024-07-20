import gradio as gr
from inference import Inference

model = Inference("1721495918", model_base="models")

def translate_text(input_text):
    """
    翻译文本
    :param input_text: 输入的文本
    :param src_lang: 源语言
    :param dest_lang: 目标语言
    :return: 翻译后的文本
    """
    translation = model.eval(input_text)
    return translation

# 定义Gradio界面
iface = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="输入要翻译的文本"),
    ],
    outputs="text",
    title="翻译应用",
    description="基于LSTM的邦布语翻译模型"
)

# 启动Gradio应用
iface.launch()
