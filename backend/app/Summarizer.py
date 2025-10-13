from sentence_transformers import SentenceTransformer#use ml model for extractive summary
#from google.cloud import storage#access documents from GCS
import spacy#sentence recognition
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration,LEDTokenizer, LEDForConditionalGeneration,PegasusTokenizer, PegasusForConditionalGeneration
import torch
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
nltk.download('punkt_tab')
#--------------------------------------------------------------------------------------------------------------------------------------------------
def extract_summary(text):
    #load  model
    extractive_model= SentenceTransformer("all-MiniLM-L6-v2")
    nlp=spacy.load("en_core_web_sm")
    doc=nlp(text)
    sentences=[sents.text for sents in doc.sents ]
    #encode indiviual sentences and entire document
    embeddings=extractive_model.encode(sentences)
    print("embeddings:",embeddings.shape)
    doc_embedding=extractive_model.encode(text)
    print("doc_embedding:",doc_embedding.shape)

    #similarity score between sentences and entire doc
    similarities=extractive_model.similarity(embeddings,doc_embedding)
    if len(sentences)>=20:
        for i in range(len(sentences)/4):
            print(similarities[i],":",sentences[i])
    elif len(sentences)>10:
        for i in range(len(sentences)/2):
            print(similarities[i],":",sentences[i])
    else:
        print("Text too short for summary")

      
def abstractive_summary(text):
    word_count = len(text.split())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Choose intermediate summarizer
    if word_count < 1500:
        print("Using Pegasus:", word_count)
        intermediate_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
        intermediate_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail").to(device)
        chunk_size = 750
    elif word_count < 3000:
        print("Using BART for intermediate + final:", word_count)
        intermediate_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        intermediate_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
        chunk_size = 1024
    else:
        print("Using AllenAI LED:", word_count)
        intermediate_tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
        intermediate_model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384").to(device)
        chunk_size = 1500

    # Step 2: Token-based chunking
    input_ids = intermediate_tokenizer.encode(text, return_tensors="pt")[0]
    chunks = []
    start_idx = 0
    while start_idx < len(input_ids):
        end_idx = min(start_idx + chunk_size, len(input_ids))
        chunks.append(input_ids[start_idx:end_idx])
        start_idx = end_idx

    # Step 3: Partial summaries
    partial_summaries = []
    for i, chunk_ids in enumerate(chunks):
        chunk_ids = chunk_ids.unsqueeze(0).to(device)
        summary_ids = intermediate_model.generate(
            chunk_ids,
            max_new_tokens=200,
            min_length=60,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=2.5,
            num_beams=4,
            early_stopping=True
        )
        summary = intermediate_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"Partial summary {i+1}:", summary)
        partial_summaries.append(summary)

    # Step 4: Use BART as final fuser
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

    combined_text = " ".join(partial_summaries)
    combined_ids = bart_tokenizer.encode(combined_text, return_tensors="pt", truncation=True).to(device)
    final_summary_ids = bart_model.generate(
        combined_ids,
        max_new_tokens=400,
        min_length=60,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=2.5,
        num_beams=4,
        early_stopping=True
    )
    final_summary = bart_tokenizer.decode(final_summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("Final Summary:", final_summary)
    return final_summary

#extract_summary("I love cats. I love dogs. I love puppies and kittens. I don't like bugs. School is fun. I want pets when i'm older. Dr. Dolittle can speak to animals")
try:
    with open("test.txt","r") as f:
        content=f.read()
    print(len(content.split()))
    abstractive_summary(content)
except Exception as e:
    print("Exception: ",e)