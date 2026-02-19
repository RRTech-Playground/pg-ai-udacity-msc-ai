import torch
from transformers import pipeline

#######################
# Device

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Device set to: {device}")


#######################
# Text Generation

prompt = "During the latest presentation OpenAI"
model = "openai-community/gpt2-large"

# Load the text generation pipeline
text_generator = pipeline("text-generation", model=model, device=device)

# Generate text
generated_texts = text_generator(prompt, max_length=100, num_return_sequences=1)

print(f"Generated text:\n{generated_texts[0]['generated_text']}")
print("\n\n")


########################################
# Greedy Search or Multinomial Sampling

for do_sample in [False, True]:  # Greedy Search  # Multinomial sampling
    generated_texts = text_generator(
        prompt, max_length=100, num_return_sequences=1, do_sample=do_sample, num_beams=1
    )

    print("-----------------------------------")
    print("Parameters:")
    print("-----------------------------------")
    print(f"do_sample={do_sample}")
    print("-----------------------------------")
    print("Generation:")
    print("-----------------------------------")
    print(generated_texts[0]["generated_text"])
    print("-----------------------------------")
    print("\n\n")

#######################
# Beam Strategy

for beams in [1, 2, 4, 8]:
    generated_texts = text_generator(
        prompt, max_length=100, num_return_sequences=1, do_sample=False, num_beams=beams
    )

    print("-----------------------------------")
    print("Parameters:")
    print("-----------------------------------")
    print(f"num_beams={beams}")
    print("-----------------------------------")
    print("Generation:")
    print("-----------------------------------")
    print(generated_texts[0]["generated_text"])
    print("-----------------------------------")
    print("\n\n")


####################################
# Beam Search Multinomial Sampling

for beams in [1, 2, 4, 8]:
    generated_texts = text_generator(
        prompt, max_length=100, num_return_sequences=1, do_sample=True, num_beams=beams
    )

    print("-----------------------------------")
    print("Parameters:")
    print("-----------------------------------")
    print(f"num_beams={beams}")
    print("-----------------------------------")
    print("Generation:")
    print("-----------------------------------")
    print(generated_texts[0]["generated_text"])
    print("-----------------------------------")
    print("\n\n")


##############################
# top-k Parameter

for k in [1, 5, 10, 50, 100, 500]:
    generated_texts = text_generator(
        prompt, max_length=100, num_return_sequences=1, top_k=k
    )

    print("-----------------------------------")
    print("Parameters:")
    print("-----------------------------------")
    print(f"top_k={k}")
    print("-----------------------------------")
    print("Generation:")
    print("-----------------------------------")
    print(generated_texts[0]["generated_text"])
    print("-----------------------------------")
    print("\n\n")


#######################
# Temperature

for temp in [0.1, 1.0, 2.0, 3.0]:
    generated_texts = text_generator(
        prompt, max_length=100, num_return_sequences=1, temperature=temp
    )

    print("-----------------------------------")
    print("Parameters:")
    print("-----------------------------------")
    print(f"temperature={temp}")
    print("-----------------------------------")
    print("Generation:")
    print("-----------------------------------")
    print(generated_texts[0]["generated_text"])
    print("-----------------------------------")
    print("\n\n")