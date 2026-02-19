from transformers import pipeline
from IPython.display import Image, display

# Text to be summarized
text = """
The Hub is home to over 5,000 datasets in more than 100 languages that can be used for a broad range of tasks across NLP, Computer Vision, and Audio. The Hub makes it simple to find, download, and upload datasets.
Datasets are accompanied by extensive documentation in the form of Dataset Cards and Dataset Preview to let you explore the data directly in your browser.
While many datasets are public, organizations and individuals can create private datasets to comply with licensing or privacy issues. You can learn more about Datasets here on Hugging Face Hub documentation.

The 🤗 datasets library allows you to programmatically interact with the datasets, so you can easily use datasets from the Hub in your projects.
With a single line of code, you can access the datasets; even if they are so large they don’t fit in your computer, you can use streaming to efficiently access the data.
"""

#summary = summarizer(text, max_length=130, min_length=30)
#print(summary[0]['summary_text'])



# Load the text generation pipeline
generator = pipeline("text-generation")

# Define the prompt for the text generation
prompt = "Scientists from University of California, Berkeley has announced that"

# Generate text based on the prompt
generated_text = generator(prompt, max_length=100, num_return_sequences=1)
print(generated_text[0]['generated_text'])


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Text to classify
text = "Transformers support framework interoperability between PyTorch, TensorFlow, and JAX. This provides the flexibility to use a different framework at each stage of a model’s life; train a model in three lines of code in one framework, and load it for inference in another."

labels = ["technology", "biology", "space", "transport"]

results = classifier(text, candidate_labels=labels, hypothesis_template="This text is about {}.")
for label, score in zip(results['labels'], results['scores']):
    print(f"{label} -> {score}")

# Replace the URL below with the URL of your image
image_url = 'https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png'

# Display the image in the notebook
#display(Image(url=image_url))

#captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")

#result = captioner(image_url)
#print(result)