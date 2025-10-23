from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# try a larger model... GPT-5. (conversation id... openai credits)
# Load the UserLM model and tokenizer
user_model_path = "microsoft/UserLM-8b"
user_tokenizer = AutoTokenizer.from_pretrained(user_model_path, trust_remote_code=True)
user_model = AutoModelForCausalLM.from_pretrained(user_model_path, trust_remote_code=True).to("cuda")

# Load the Qwen model and tokenizer
qwen_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_path,
    torch_dtype="auto",
    device_map="auto"
)
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)

# Initialize conversation history
conversation_history = [
    {"role": "system", "content": "You are a user talking to a language model to learn how to bake cookies."}
]

# UserLM tokens
end_token = "<|eot_id|>"
end_token_id = user_tokenizer.encode(end_token, add_special_tokens=False)

end_conv_token = "<|endconversation|>"
end_conv_token_id = user_tokenizer.encode(end_conv_token, add_special_tokens=False)

# Number of conversation turns
num_turns = 4

for turn in range(num_turns):
    print(f"\n{'='*60}")
    print(f"Turn {turn + 1}")
    print('='*60)

    # UserLM generates user message
    print("\n[UserLM generating...]")
    # Give UserLM the full conversation history
    user_messages = [conversation_history[0]] + conversation_history[1:]
    print(user_messages)

    user_inputs = user_tokenizer.apply_chat_template(user_messages, return_tensors="pt").to("cuda")
    user_outputs = user_model.generate(
        input_ids=user_inputs,
        do_sample=True,
        top_p=0.8,
        temperature=1,
        max_new_tokens=100,
        eos_token_id=end_token_id,
        pad_token_id=user_tokenizer.eos_token_id,
        bad_words_ids=[[token_id] for token_id in end_conv_token_id]
    )
    user_response = user_tokenizer.decode(user_outputs[0][user_inputs.shape[1]:], skip_special_tokens=True)
    print(f"User: {user_response}")

    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_response})

    # Qwen generates assistant response
    print("\n[Qwen generating...]")
    # Prepare messages for Qwen (exclude the system prompt meant for UserLM)
    qwen_messages = [
        {"role": "system", "content": "You are Qwen, a helpful AI assistant whose job is to converse with the user."}
    ]
    # Add the conversation history (skip the first system message)
    for msg in conversation_history[1:]:
        qwen_messages.append(msg)

    text = qwen_tokenizer.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    qwen_model_inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)

    qwen_generated_ids = qwen_model.generate(
        **qwen_model_inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.8,
        temperature=1
    )
    qwen_generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(qwen_model_inputs.input_ids, qwen_generated_ids)
    ]

    qwen_response = qwen_tokenizer.batch_decode(qwen_generated_ids, skip_special_tokens=True)[0]
    print(f"Assistant: {qwen_response}")

    # Add assistant message to conversation history
    conversation_history.append({"role": "assistant", "content": qwen_response})

print("\n" + "="*60)
print("Conversation complete!")
print("="*60)