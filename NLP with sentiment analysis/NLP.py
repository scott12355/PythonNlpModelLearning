from transformers import pipeline, TFAutoModelForSeq2SeqLM, AutoTokenizer
from os import system
import torch, os

# Check if CUDA is available (i.e., a GPU is present)
# Check for MPS (Apple Silicon) support
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)
# Load the Hugging Face model
# Load the Hugging Face model
# Load the Hugging Face tokenizer        EleutherAI/gpt-neo-1.3B
modelName = "Qwen/Qwen2.5-0.5B-Instruct" #Qwen/Qwen2.5-1.5B-Instruct 
chatbot_model = pipeline("text-generation", model=modelName, device=device, batch_size=16)
conversation_history = []
sentiment_model = pipeline("sentiment-analysis", device=device)



def generate_response(user_input, sentiment):
    if sentiment['label'] == 'NEGATIVE':
        prompt = f"The user is upset, respond with empathy and support: {user_input}"
    else:
        prompt = f"Respond to the following query: {user_input}"

    conversation_history.append(
        {"role": "user", "content": prompt},
    )
    result = chatbot_model(conversation_history, num_return_sequences=1, max_new_tokens=250)
    conversation_history.append(
        {"role": "assistant", "content": result[0]['generated_text'][-1]['content']},
    )
    print(prompt)
    return result[0]['generated_text'][-1]['content'] 
   
quit = False
while quit == False:
    user_input = input("Enter your message\n")
    

    if user_input.lower() == "quit()":
        quit = True
        break
    print("Thinking...\n")
    sentiment = sentiment_model(user_input)[0]
    print(sentiment)
    response = generate_response(user_input, sentiment)
    # system('clear')
    print(response)

file_path = "logs/conversation_history.txt"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, 'w') as file:
    file.write("\nConversation History:\n")
    # Write each element to the file on a new line
    for item in conversation_history:
        file.write(f"{item}\n")