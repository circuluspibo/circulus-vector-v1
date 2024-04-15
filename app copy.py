import random
import gradio as gr
import ctranslate2
#import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download, hf_hub_download
from llama_cpp import Llama

model_en2ko = ctranslate2.Translator(snapshot_download(repo_id="circulus/canvers-en2ko-ct2-v1"), device="cpu")
token_en2ko = AutoTokenizer.from_pretrained("circulus/canvers-en2ko-v1")

model_ko2en = ctranslate2.Translator(snapshot_download(repo_id="circulus/canvers-ko2en-ct2-v1"), device="cpu")
token_ko2en = AutoTokenizer.from_pretrained("circulus/canvers-ko2en-v1")

def trans_ko2en(prompt):
  source = token_ko2en.convert_ids_to_tokens(token_ko2en.encode(prompt))
  results = model_ko2en.translate_batch([source])
  target = results[0].hypotheses[0]
  return token_ko2en.decode(token_ko2en.convert_tokens_to_ids(target), skip_special_tokens=True)

def trans_en2ko(prompt):
  source = token_en2ko.convert_ids_to_tokens(token_en2ko.encode(prompt))
  results = model_en2ko.translate_batch([source])
  target = results[0].hypotheses[0]
  return token_en2ko.decode(token_en2ko.convert_tokens_to_ids(target), skip_special_tokens=True)

llm2 = Llama(model_path=hf_hub_download(repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"), chat_format="chatml")

type = "You are to roleplay as David from genius boy following instructions with images extremely well. Help as much as you can. You gives helpful, detailed, and polite answer to human's question."

def chat(message, history):
  print("start")
  sentence = ""
  prompt = trans_ko2en(message)
  print('trans', prompt)
  streamer = llm2.create_chat_completion(
      messages = [
        {"role": "system", "content": f"{type}"},
        {"role": "user", "content": f"{prompt}"}
      ],
      stream=True,
  )

  for new_text in streamer:
    print(new_text)
    new_text = new_text["choices"][0]["delta"].get("content")

    if new_text != None:
      if new_text.startswith("###") or new_text.startswith("Assistant:") or new_text.startswith("User:") or new_text.startswith("<|user|>") or new_text.startswith("<|im_start|>assistant"):  #User is stop keyword
        print("skipped",new_text)
      else:
        if new_text.find("\n") > -1:
          if len(sentence) > 3:
            result = trans_en2ko(sentence + new_text) #.replace("\n","")
            print(sentence + new_text ,result)
            sentence = ""
            if new_text.find("\n\n") > -1:
              yield result + "\n\n" 
            else:
              yield result  + "\n"
          else:
            yield new_text
        elif new_text.find(".") > -1 and len(sentence) > 3:
          result = trans_en2ko(sentence + new_text)
          print(sentence + new_text ,result)
          sentence = ""
          if new_text.find("\n") > -1:
            yield result + "\n"
          elif result.find(".") > -1:
            yield result + " "
          else:
            yield result + ". "
        else:
          sentence = sentence + new_text
  print("end")

desc = """
# Welcome to chatbot test
<table>
  <tr>
    <td><img width="256" height="256" src='https://canvers.net/v1/v/media/65c378439469a7872e13888e'/></td>
    <td><video width="512" height="512" controls  src="https://www.canvers.net/image/demo.mp4" autoplay></video></td>
  </tr>
</table>
"""

demo = gr.ChatInterface(fn=chat, description=desc, fill_height=True)

if __name__ == "__main__":
    demo.launch()
