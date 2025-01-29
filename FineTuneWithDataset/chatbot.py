from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from flask import Flask, request, jsonify
from transformers import pipeline
import torch

# Initialize Flask
app = Flask(__name__)
import tensorflow as tf

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)
        
# Initialize conversation history
conversation_history = []
sentiment_analysis = pipeline("sentiment-analysis", device=1)
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

chatbot_model = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, batch_size=16)


def adjust_prompt_based_on_sentiment(prompt):
    sentiment = sentiment_analysis(prompt)[0]
    if sentiment['label'] == 'NEGATIVE':
        prompt = f"User seems upset. Respond in a calming manner: {prompt}"
    return prompt

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    conversation_history.append({"role": "user", "content": user_input})

    # Create a conversation string by concatenating user and assistant turns
    prompt = ""
    for turn in conversation_history:
        prompt += f"{turn['role']}: {turn['content']}\n"
    print(prompt)
    result = chatbot_model(prompt, num_return_sequences=1, max_new_tokens=250)
    #response_text = result[0]['generated_text'][-1]['content'] 

    #conversation_history.append({"role": "assistant", "content": response_text})
    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
