
import streamlit as st

import deepl
auth_key = "9d5d6377-86f6-5862-90e2-783b691526a2:fx" 
translator = deepl.Translator(auth_key)

from summarizer import Summarizer



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
  model = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)
  result = model(body, num_sentences=4)
  final_text = translator.translate_text(result, target_lang="IT")
  return final_text.text

@st.cache #decorator
def translate3(text):
  new_text = translator.translate_text(text, target_lang="EN-GB")
  body = new_text.text
  model = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)
  result = model(body, num_sentences=5)
  final_text = translator.translate_text(result, target_lang="IT")
  return final_text.text

@st.cache #decorator
def translate4(text):
  new_text = translator.translate_text(text, target_lang="EN-GB")
  body = new_text.text
  model = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)
  result = model(body, num_sentences=6)
  final_text = translator.translate_text(result, target_lang="IT")
  return final_text.text



  

def main():
	

	# Title
	st.title("Demo2 di News Summarizer")
	st.subheader("Riassumere testi in pochi semplici click")
	

	# Summarization
	if st.checkbox("Get the summary of your text"):
		st.subheader("Sto lavorando anche con altri modelli, per il momento condivido questa demo con BERT ")

		message = st.text_area("Enter Text","Type Here....")
		summary_options = st.selectbox("Choose Summarizer",['bert n.periodi:3','bert n.periodi:4', 'bert n.periodi:5', 'bert n.periodi:6', 'bert n.periodi:2'])
		if st.button("Summarize"):
			if summary_options == 'bert n.periodi:3':
				st.text("Using bert n.periodi:3 ..")
				summary_result = translate(message)
				
				
			elif summary_options == 'bert n.periodi:4':
				st.text("Using bert n.periodi:4 ..")
				summary_result = translate2(message)
			
			elif summary_options == 'bert n.periodi:5':
				st.text("Using bert n.periodi:5 ..")
				summary_result = translate3(message)
				
			elif summary_options == 'bert n.periodi:6':
				st.text("Using bert n.periodi:6 ..")
				summary_result = translate4(message)
				
			else:
				st.warning("Using Default Summarizer")
				st.text("Using Bert Summarizer ..")
				summary_result = translate(message)
			st.success(summary_result)

if __name__ == '__main__':
	main()
	
