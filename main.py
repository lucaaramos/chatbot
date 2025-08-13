from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Definir token de padding
tokenizer.pad_token = tokenizer.eos_token

def generate_response(conversation_history, max_length=200):
    inputs = tokenizer(
        conversation_history,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. Bucle interactivo
print("English GPT-2 Chatbot activated. Type 'quit' to exit.")
conversation_history = ""

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break

        conversation_history += f"User: {user_input}\nBot:"

        full_response = generate_response(conversation_history)
        
        bot_reply = full_response.split("Bot:")[-1].strip()

        conversation_history += f" {bot_reply}\n"

        print("Bot:", bot_reply)

    except KeyboardInterrupt:
        print("\n(Type 'quit' to exit or continue chatting)")
