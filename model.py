from jiwer import wer

from libs.ai.openai import ICNAzureChatOpenAI
from langchain.prompts import (
	HumanMessagePromptTemplate,
)
from sacrebleu.metrics import BLEU


class MachineTranslation:

	def __init__(self, data_dir: str, output_dir: str):
		self.chat = ICNAzureChatOpenAI()
		self.bleu = BLEU()
		self.from_path = data_dir
		self.to_path = output_dir

	def translate(self, text: str, lang_to: str) -> str:
		"""
		Translates the text into the language lang_to
		"""

		template_string = """Translate the text
			into the language {style}. Let's think through it step by step. Follow these rules: \ 
			
			Keep all formatting and meaning. 
			The context is private equity and markets. The audience is for investors and clients 
			familiar with financial terms and concepts. 
			Make accurate translations using the context and audience.
			If there is no text to translate, return the text inputted. \
			text: {context}
			"""

		text = text.rstrip()
		if text.isdigit():
			return text

		else:
			prompt_template = HumanMessagePromptTemplate.from_template(template_string)
			msg = prompt_template.format(
				style=lang_to,
				context=text)
			response = self.chat([msg])
			return response.content

	def evaluate_bleu(self, hyp: list, ref: list[list]) -> float:
		"""
		Hyp is list, ref is list of lists
		Returns BLEU score of corpus
		"""
		s = self.bleu.corpus_score(hyp, ref)
		return s.score

	def evaluate_wer(self, hyp: list, ref: list[list]):
		"""
		Hyp is list, ref is list of lists
		Returns Word error rate
		"""
		return wer(ref, hyp)
