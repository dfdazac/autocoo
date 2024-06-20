import os.path as osp
from argparse import ArgumentParser

import pandas as pd
import torch
import transformers
from tqdm import tqdm


SYN_COL = 'Synonyms'


def clean_synonyms_file(csv_file: str, prompt_file: str):
    directory = osp.dirname(csv_file)
    output_fname = 'synonyms_clean_results.csv'
    df = pd.read_csv(csv_file)

    # The Synonyms columns contains a list of strings but it's read as a string
    # We convert it here.
    df[SYN_COL] = df[SYN_COL].apply(eval)

    # LLM instantiation
    model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.float16},
        device='cuda'
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    def filter_row(row: pd.Series, prompt_prefix: str) -> list[str]:
        """Filter rows based on the response of an LLM to the prompt."""
        result = []
        for term in row[SYN_COL]:
            # Prompt LLM for relevance of the term
            messages = [{"role": "user",
                         "content": f'{prompt_prefix} {term}'}]
            prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)
            outputs = pipeline(prompt,
                               return_full_text=False,
                               max_new_tokens=500,
                               do_sample=True,
                               temperature=0.7,
                               top_k=50,
                               top_p=0.95,
                               eos_token_id=terminators,
                               pad_token_id=pipeline.model.config.eos_token_id)
            outputs = outputs[0]['generated_text'].lower()

            if '[yes]' in outputs:
                result.append(term)

        return result

    with open(args.prompt_file) as f:
        prompt_prefix = f.read()

    # Use LLM to drop non-relevant items, and drop rows where result is empty
    tqdm.pandas(desc='Filtering synonyms')
    df[SYN_COL] = df.progress_apply(filter_row,
                                    axis=1,
                                    prompt_prefix=prompt_prefix)
    df = df[df[SYN_COL].apply(len) > 0]

    df = df.reset_index(drop=True)
    df.to_csv(osp.join(directory, output_fname), index=False)


if __name__ == "__main__":
    parser = ArgumentParser('Cleans file with ')
    parser.add_argument('--csv_file',
                        help='Synonyms file with terms to filter')
    parser.add_argument('--prompt_file',
                        help='Plain text file with prompt to use')
    args = parser.parse_args()

    clean_synonyms_file(args.csv_file, args.prompt_file)



