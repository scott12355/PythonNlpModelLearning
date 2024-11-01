from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import os

app = Flask(__name__)

# Check for MPS (Apple Silicon) support
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load the Hugging Face models
model_name = "Qwen/Qwen2.5-3B-Instruct"  # or any other model suitable for your needs
chatbot_model = pipeline("text-generation", model=model_name, device=device, batch_size=8)
sentiment_model = pipeline("sentiment-analysis", device=device)

# Store conversation history
conversation_history = []

def generate_response(user_input, sentiment):
    if sentiment['label'] == 'NEGATIVE':
        prompt = f"The user is upset, respond with empathy and support: {user_input}"
    else:
        prompt = f"Respond to the following query: {user_input}"

    # Append the user prompt to the conversation history
    conversation_history.append({"role": "user", "content": prompt})

    # Generate response from the model
    result = chatbot_model(conversation_history, num_return_sequences=1, max_new_tokens=250)

    # Append the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": result[0]['generated_text'][-1]['content'] })

    return result[0]['generated_text'][-1]['content'] 


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({'error': 'Please provide a message.'}), 400

    sentiment = sentiment_model(user_input)[0]
    response = generate_response(user_input, sentiment)
    

    return jsonify({'response': response , 'sentiment': sentiment})

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(conversation_history)

if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    file_path = "logs/conversation_history.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write the conversation history to a file on startup
    with open(file_path, 'w') as file:
        file.write("\nConversation History:\n")
        for item in conversation_history:
            file.write(f"{item}\n")

    app.run(host='0.0.0.0', port=5500)
