import torch, os
from transformers import pipeline
from os import system



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
chatbot_model = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct",trust_remote_code=True, device=device, batch_size=16)
conversation_history = []

def generate_response(user_input):
    # Generate a response using the Hugging Face model
    conversation_history.append(
        {"role": "user", "content": user_input},
    )
    result = chatbot_model(conversation_history, num_return_sequences=1, max_new_tokens=250)
    conversation_history.append(
        {"role": "assistant", "content": result[0]['generated_text'][-1]['content']},
    )
    return result[0]['generated_text'][-1]['content'] 

quit = False
while quit == False:
    question = input("Enter your message\n")
    if question.lower() == "quit()":
        quit = True
        break
    print("Thinking...\n")
    response = generate_response(question)
    system('clear')
    print(response)
    
file_path = "logs/conversation_history.txt"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, 'w') as file:
    file.write("\nConversation History:\n")
    # Write each element to the file on a new line
    for item in conversation_history:
        file.write(f"{item}\n")