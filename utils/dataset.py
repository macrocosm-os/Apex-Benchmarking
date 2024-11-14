import random
from datasets import load_dataset

class DatasetWrapper:
    def __init__(self, name: str = 'mmlu', batch_size: int = 10, seed: int = 42) -> None:
        self.name = name
        self.batch_size = batch_size
        random.seed(seed)
        if name == 'mmlu':
            self.dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train", streaming=True)
            self.dataset_size = 99842
            self.process_function = self.process_mmlu
        
        self.batch = self.get_random_batch()

    def process_mmlu(self, example):
        one_shot = """
[Example 1]
What is the capital of Texas?
A. Paris
B. London
C. Austin
D. Houston
Answer: C 
"""
        return {'topic': example['subject'],
                'query': example['question'], 
                'options': example['choices'],
                'answer_idx': example['answer'],
                'answer': example['choices'][example['answer']],
                'formatted_options' : [f"{chr(65+i)}. {choice}" for i, choice in enumerate(example['choices'])],
                'preface': one_shot,
                }

    def get_random_batch(self):
        skip_count = random.randint(0, self.dataset_size - self.batch_size)
        dataset_iter = iter(self.dataset)
        for _ in range(skip_count):
            next(dataset_iter, None)
        
        return iter([next(dataset_iter) for _ in range(self.batch_size)])
    
    def __next__(self):
        try:
            return self.process_function(next(self.batch))
        except StopIteration:
            self.batch = self.get_random_batch()
            return self.process_function(next(self.batch))
        
def create_challenge(context: dict) -> str:
    formatted_options = "\n".join(context['formatted_options'])
    challenge = f"""{context['preface']}
[Input Question]\n{context['query']}
{formatted_options}
Answer:"""
    return challenge
