import streamlit as st
import deepl
from summarizer import Summarizer
from dataclasses import dataclass
from typing import Optional

@dataclass
class TranslationConfig:
    """Configuration for DeepL translation"""
    auth_key: str = "9d5d6377-86f6-5862-90e2-783b691526a2:fx"
    source_lang: str = "IT"
    intermediate_lang: str = "EN-GB"
    target_lang: str = "IT"

class TextProcessor:
    """Handles text translation and summarization"""
    
    def __init__(self, config: TranslationConfig):
        self.translator = deepl.Translator(config.auth_key)
        self.summarizer = Summarizer(
            'distilbert-base-uncased',
            hidden=[-1, -2],
            hidden_concat=True
        )
        self.config = config

    @st.cache(show_spinner=False)
    def process_text(self, text: str, num_sentences: int) -> str:
        """
        Translate text to English, summarize it, and translate back to Italian
        
        Args:
            text: Input text to process
            num_sentences: Number of sentences for summary
            
        Returns:
            Translated and summarized text
        """
        # Translate to English
        english_text = self.translator.translate_text(
            text,
            target_lang=self.config.intermediate_lang
        ).text
        
        # Summarize
        summary = self.summarizer(
            english_text,
            num_sentences=num_sentences
        )
        
        # Translate back to Italian
        final_text = self.translator.translate_text(
            summary,
            target_lang=self.config.target_lang
        )
        
        return final_text.text

class SummarizerApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.config = TranslationConfig()
        self.processor = TextProcessor(self.config)
        self.summary_options = {
            'bert n.periodi:3': 3,
            'bert n.periodi:4': 4,
            'bert n.periodi:5': 5,
            'bert n.periodi:6': 6
        }

    def render_header(self):
        """Render application header"""
        st.title("Demo2 di News Summarizer")
        st.subheader("Riassumere testi in pochi semplici click")

    def render_main_content(self):
        """Render main application content"""
        if not st.checkbox("Get the summary of your text"):
            return

        st.subheader(
            "In questa demo sono state utilizzate diversi parametri per il "
            "modello BERT. N.periodi corrisponde all'originale num_sentences, "
            "la quale consente di scegliere la lunghezza del riassunto finale."
        )

        # Text input
        message = st.text_area("Enter Text", "Type Here....")
        
        # Summary options
        selected_option = st.selectbox(
            "Choose Summarizer",
            list(self.summary_options.keys())
        )

        if st.button("Summarize"):
            self.process_summary(message, selected_option)

    def process_summary(self, text: str, option: str):
        """Process and display summary"""
        st.text(f"Using {option} ..")
        num_sentences = self.summary_options.get(option, 3)  # Default to 3 if option not found
        
        try:
            summary_result = self.processor.process_text(text, num_sentences)
            st.success(summary_result)
        except Exception as e:
            st.error(f"Error processing text: {str(e)}")

    def run(self):
        """Run the Streamlit application"""
        self.render_header()
        self.render_main_content()

def main():
    app = SummarizerApp()
    app.run()

if __name__ == '__main__':
    main()
