Project Title: Training a Small Language Model From Scratch


1. System Overview:
Describe how your system works as a pipeline.
Our system works like a simple pipeline. First, we load text data from SlimPajama using streaming so we don’t need to download the whole dataset. Then we clean and filter the text by removing empty or very short samples and obvious junk. Next, we tokenize the text using an existing tokenizer, which turns the text into token IDs. After that, we pack the tokens into 512-token chunks so the model trains efficiently. We then train a ~300M-parameter transformer to predict the next token, and we evaluate it during training by tracking validation loss. Finally, we generate text by giving the trained model a prompt and letting it continue the text.


2. Dataset Description
Dataset name and source
SlimPajama (a mix of web crawler text, Wikipedia, and StackExchange). Source: https://huggingface.co/datasets/DKYoon/SlimPajama-6B