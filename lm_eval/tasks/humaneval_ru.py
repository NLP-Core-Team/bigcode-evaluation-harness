import re

from evaluate import load

from lm_eval.base import Task
from lm_eval.utils import extract_generation_code

_CITATION = """
"""


class HumanEvalRu(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "NLPCoreTeam/humaneval_ru"

    def __init__(self, prompt=""):
        self.counter = 0
        self.prompt = prompt
        if self.prompt in ['instruct','instruct_mistral','instruct_deepseek']:
            stop_words=["<|endoftext|>", "<extra_id_0>","</s>","<|/code|>","<|EOT|>"]
        else:
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<|endoftext|>", "<extra_id_0>", "<|/code|>"]
        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["train"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.prompt == "mistral":
            prompt = f"[INST] {doc['prompt'].strip()} [/INST]"
        elif self.prompt == "markdown":
            prompt = f"```python\n{doc['prompt'].strip()}"
        elif self.prompt == "github":
            prompt = f"<|code|>\n{doc['prompt'].strip()}"
        elif self.prompt == "instruct":
            prompt = f"Пожалуйста, допиши код функции до конца. Нельзя вносить изменения в код, можно только дописывать. Пожалуйста, верни всю завершенную функцию в блоке кода. Вот код, который нужно закончить:\n```python\n{doc['prompt'].strip()}\n```"
        elif self.prompt == "instruct_mistral":
            prompt = f"[INST] Пожалуйста, допиши код функции до конца. Нельзя вносить изменения в код, можно только дописывать. Пожалуйста, верни всю завершенную функцию в блоке кода. Вот код, который нужно закончить:\n```python\n{doc['prompt'].strip()}\n``` [/INST]"
        elif self.prompt == "instruct_deepseek":
            prompt = f"<｜begin▁of▁sentence｜>You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\nПожалуйста, допиши код функции до конца. Нельзя вносить изменения в код, можно только дописывать. Пожалуйста, верни всю завершенную функцию в блоке кода. Вот код, который нужно закончить:\n```python\n{doc['prompt'].strip()}\n```### Response:\n"
        else:
            prompt = doc["prompt"].strip()
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_dataset()[idx]["prompt"].strip()
        if self.prompt in ['instruct','instruct_mistral','instruct_deepseek']:
            return extract_generation_code(task_id=idx, output=generation, prompt=prompt,lang_code='python')
        else:
            return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("code_eval")
        results, _ = code_metric.compute(
            references=references,
            predictions=generations,
        )
        return results
