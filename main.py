from lm_eval import tasks
from lm_eval.utils import TokenizedDataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser, CodeLlamaTokenizer
from torch.utils.data.dataloader import DataLoader
import torch
from lm_eval.arguments import EvalArguments
from lm_eval.evaluator import Evaluator
from lm_eval.tasks import ALL_TASKS
import fnmatch
import datasets
import transformers
from functools import partial
import multiprocessing
import torch.multiprocessing as mp
import json
import os
import random
import numpy as np
import random
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
def seed_everything(seed=42):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model_path",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--modeltype",
        default="causal",
        help="AutoModel to use, it can be causal or seq2seq",
    )
    parser.add_argument(    
        "--peft_model",
        type=str,
        default=None,
        help="Adapter to the PEFT base model. Can be utilized for loading PEFT adapters such as a LoRA trained model. The --model parameter needs to be the base model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="Optional offset to start from when limiting the number of samples",
    )    
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )    
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompt",
        help="Prompt type to use for generation in HumanEvalPack tasks",
    )
    parser.add_argument("--max_memory_per_gpu", type=str, default=None)
    parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
    )
    parser.add_argument('-d', '--devices', type=str, default='0')
    parser.add_argument('--instance_per_device', type=int, default='1')
    return parser.parse_args()


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory

def get_dataset_list(dataset, devices, instance_per_device):
    random.shuffle(dataset)
    num_process = devices * instance_per_device
    size = len(dataset)//num_process
    reminder = len(dataset)%num_process
    dataset_lists = list()
    for i in range(num_process):
        dataset_lists.append(
            dataset[i*size:(i+1)*size]
        )
    dataset_reminder = len(dataset) - size * num_process
    dataset_reminder = len(dataset) - dataset_reminder
    for i in range(reminder):
        dataset_lists[i].append(
            dataset[dataset_reminder+i]
        )
    return dataset_lists

def run_job(dataset, task, gpu_device, args, return_dict):
    tokenizer = AutoTokenizer.from_pretrained(
                    args.model_path,
                    use_fast=False,
                    truncation_side="left",
                    padding_side="right", # padding on the right is needed to cut off padding in `complete_code`
                )
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    try:
        tokenizer.pad_token = tokenizer.eos_token
    # Some models like CodeGeeX2 have pad_token as a read-only property
    except AttributeError:
        print("Not setting pad_token to eos_token")
        pass
    # tokenizer.eos_token = tokenizer.bos_token
    model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    use_flash_attention_2=True,
                )
    model.to(torch.device(gpu_device))
    evaluator = Evaluator(model, tokenizer, args)
    generations, references = evaluator.generate_text(task, dataset)
    process_name = multiprocessing.current_process().name
    return_dict[process_name] = {
        'generations': generations,
        'references': references,
    }

def main():
    args = parse_args()
    seed_everything(seed=args.seed)

    if Path(args.metric_output_path).exists():
        print('Metrics already calculated')
        with open(args.metric_output_path) as f:
            print(f.read())
        exit()

    if not Path(args.model_path).exists():
        print('Model path is incorrect')
        exit()

    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    
    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)
    results = {}

    #model_path = '/home/jovyan/chernysh/instruct/output/instruct_v0.4/checkpoint-339'
    
    for task_name in task_names:
        task = tasks.get_task(task_name, args)
        dataset = task.get_dataset()
        dataset_list = list(dataset)
        for i, item in enumerate(dataset_list):
            item['idx'] = i

        devices = [f'cuda:{idx}' for idx in args.devices.split(',')]

        datasets_list = get_dataset_list(dataset_list, len(devices), args.instance_per_device)
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = list()
        for device_idx, device in enumerate(devices):
            for i in range(args.instance_per_device):
                idx = i+device_idx*args.instance_per_device
                partial_dataset = datasets_list[idx]
                torch.cuda.empty_cache()
                func = partial(
                    run_job,
                    dataset=partial_dataset,
                    task=task,
                    gpu_device=device,
                    args=args,
                    return_dict=return_dict,
                )

                processes.append(
                    mp.Process(
                        name=f'{idx}', 
                        target=func,
                        daemon=True
                    )
                )

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        gen_results = {
            'generations': [],
            'references': [],
        }
        for key in list(gen_results.keys()):
            for i in list(return_dict.values()):
                gen_results[key] += i[key]

        if args.save_generations:
            Path(args.save_generations_path).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_generations_path, "w", encoding='utf8') as fp:
                json.dump(gen_results['generations'], fp)
                print(
                    f"generations were saved at {args.save_generations_path}"
                )
        if args.save_references:
            with open("references.json", "w", encoding='utf8') as fp:
                json.dump(gen_results['references'], fp)
                print("references were saved at references.json")

        if not args.generation_only:
            print("Evaluating generations...")
            results = {}
            results[task_name] = task.process_results(gen_results['generations'], gen_results['references'])
            results["config"] = vars(args)

            dumped = json.dumps(results, indent=2)
            print(dumped)
            Path(args.metric_output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(args.metric_output_path, "w", encoding='utf8') as f:
                f.write(dumped)

if __name__ == '__main__':
    main()

