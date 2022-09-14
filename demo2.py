
import streamlit as st

import deepl
auth_key = "9d5d6377-86f6-5862-90e2-783b691526a2:fx" 
translator = deepl.Translator(auth_key)

from summarizer import Summarizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
#from transformers import pipeline



@st.cache #decorator
def translate(text):
  new_text = translator.translate_text(text, target_lang="EN-GB")
  body = new_text.text
  model = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)
  result = model(body, num_sentences=3)
  final_text = translator.translate_text(result, target_lang="IT")
  return final_text.text

@st.cache #decorator
def translate2(text):
  new_text = translator.translate_text(text, target_lang="EN-GB")
  body = new_text.text
  tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
  model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
  tokens_input = tokenizer.encode("summarize: "+body, return_tensors='pt', max_length=1024, truncation=True)
  ids = model.generate(tokens_input, min_length=150, max_length=180)
  summary = tokenizer.decode(ids[0], skip_special_tokens=True)
  final_text = translator.translate_text(summary, target_lang="IT")
  return final_text.text



  

def main():
	

	# Title
	st.title("Demo2 di News Summarizer")
	st.subheader("Riassumere testi in pochi semplici click")
	

	# Summarization
	if st.checkbox("Get the summary of your text"):
		st.subheader("Sto lavorando anche con altri modelli, per il momento condivido questa demo con BERT ")

		message = st.text_area("Enter Text","Type Here....")
		summary_options = st.selectbox("Choose Summarizer",['bert','Bart-facebook'])
		if st.button("Summarize"):
			if summary_options == 'bert':
				st.text("Using Bert Summarizer ..")
				summary_result = translate(message)
				
				
			elif summary_options == 'Bart-facebook':
				st.text("Using Bart-facebook Summarizer ..")
				summary_result = translate2(message)
			
			#elif summary_options == 'pipeline api':
				#st.text("Using Pipeline API Summarizer ..")
				#summary_result = translate3(message)
				
			#elif summary_options == 't-5 large':
				#st.text("Using t-5 Summarizer ..")
				#summary_result = translate4(message)
				
			else:
				st.warning("Using Default Summarizer")
				st.text("Using Bert Summarizer ..")
				summary_result = translate2(message)
			st.success(summary_result)

if __name__ == '__main__':
	main()
	
