import gradio as gr
import requests
import json


with open('config.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

print(data)

type = data['type']
url = data['url']

def chat(message, history):

  txt = ""

  print(message)
  #r = requests.post('https://oe-napi.circul.us/v1/txt2chat', json = { "body" : {'prompt ': message, 'history' : [''], 'lang' : 'ko','type':'assist', 'rag' : '','temp' : 1}, 'prompt ': message, 'history' : [''], 'lang' : 'ko','type':'assist', 'rag' : '','temp' : 1}, stream=True)
  r = requests.post('http://222.112.0.215:59522/v1/chat', json={ 'prompt': message, 'history' : [''], 'lang' : 'ko','type': type, 'rag' : '','temp' : 1 }, stream=True)

  print(r)
  for line in r.iter_lines():
    line = line.decode('utf-8')
    print(line)
    txt = txt + "\n" + line
    yield txt




desc = f'<video width="512" height="512" controls src="{url}" autoplay></video>'
# http://222.112.0.215:59522/v1/v/media/658aaf340833518cf6140dd8?type=mp4&length=158033

demo = gr.ChatInterface(fn=chat, description=desc, fill_height=True)

if __name__ == "__main__":
    demo.launch()
