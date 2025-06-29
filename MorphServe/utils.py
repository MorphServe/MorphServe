import evaluate
import numpy as np

def cdiv(a: int, b: int):
    return (a + b - 1) // b

KB = 1024
MB = 1024*1024
GB = 1024*1024*1024
TB = 1024*1024*1024*1024

def create_request_metrics_dict(request):
    """Convert request metrics to a dictionary format for CSV"""
    metrics = {
        'prompt_len': request.prompt_len,
        'output_len': request.output_len,

        'arrival_time': request.req_metrics.arrival_time,
        'tokenized_time': request.req_metrics.tokenized_time,
        'queued_time': request.req_metrics.queued_time,
        'first_token_time': request.req_metrics.first_token_time,
        'completion_time': request.req_metrics.completion_time,
        
        'time_tokenized': request.req_metrics.tokenized_time - request.req_metrics.arrival_time,
        'time_queued': request.req_metrics.queued_time - request.req_metrics.tokenized_time,
        'time_to_first_token': request.req_metrics.first_token_time - request.req_metrics.tokenized_time,
        'time_per_output_token': (request.req_metrics.completion_time - request.req_metrics.first_token_time) / (request.output_len - 1) if request.output_len > 1 else 0,
        'total_inference_time': request.req_metrics.completion_time - request.req_metrics.arrival_time
    }
    return metrics

def trim_tokens(df, context_col="ContextTokens", gen_col="GeneratedTokens", max_total=4090, gen_threshold=64):
    df = df.copy()
    
    total_tokens = df[context_col] + df[gen_col]

    # Case 1: where total >= 4096 and gen_tokens <= 64
    cond_case1 = (total_tokens >= max_total) & (df[gen_col] <= gen_threshold)
    df.loc[cond_case1, context_col] = max_total - df.loc[cond_case1, gen_col]
    
    # Case 2: where total >= 4096 and gen_tokens > 64
    cond_case2 = (total_tokens >= max_total) & (df[gen_col] > gen_threshold)
    ratio = max_total / (df.loc[cond_case2, context_col] + df.loc[cond_case2, gen_col])
    df.loc[cond_case2, context_col] = (df.loc[cond_case2, context_col] * ratio).astype(int)
    df.loc[cond_case2, gen_col]     = (df.loc[cond_case2, gen_col] * ratio).astype(int)

    return df

def tokenize_tokens(df, tokenizer, context_col="ContextTokens"):
    prompts = ["Paris is the capital city of"]
    tokenizer.pad_token = tokenizer.eos_token
    df = df.copy()
    input_ids_trace = list()
    for i, row in df.iterrows():
        input_ids = tokenizer(prompts, padding="max_length",
                      max_length=row[context_col]).input_ids
        tokenized_ids = input_ids[0]
        input_ids_trace.append(tokenized_ids)
        # print(f"i: {i}, context_len: {row[context_col]}, tokenized_ids_len: {len(tokenized_ids)}")
    return input_ids_trace

def tokenize_long_bench(dataset, tokenizer, prompt_max_len, reference_max_len):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    def tokenize_fn(example):
        # Tokenize input (prompt) with truncation only
        prompt_ids = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=prompt_max_len,
            padding="max_length",
            return_tensors=None
        )

        # Tokenize output (reference) with truncation only
        # reference_ids = tokenizer(
        #     example["reference"],
        #     truncation=True,
        #     max_length=reference_max_len,
        #     return_tensors=None
        # )
        
        # Tokenize output with truncation and padding
        reference_ids = tokenizer(
            example["reference"],
            truncation=True,
            max_length=reference_max_len,
            padding="max_length",
            return_tensors=None
        )

        return {
            "input_ids": prompt_ids["input_ids"],
            "attention_mask": prompt_ids["attention_mask"],
            "labels": reference_ids["input_ids"],
        }

    return dataset.map(tokenize_fn, batched=False, remove_columns=dataset.column_names)


def F1_calculate(predictions, references):
    f1_list = []
    assert len(predictions) == len(references)
    for pred_tokens, ref_tokens in zip(predictions, references):
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        if not pred_set or not ref_set:
            f1 = 0.0
        else:
            overlap = pred_set & ref_set
            precision = len(overlap) / len(pred_set)
            recall = len(overlap) / len(ref_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_list.append(f1)
    return f1_list

def rouge_calculate(predictions, references):
    rouge = evaluate.load("rouge")
    assert len(predictions) == len(references)
    org_rouge1_score_list = []
    org_rouge2_score_list = []
    org_rougeL_score_list = []
    org_rougeLsum_score_list = []
    for i in range(len(predictions)):
        rouge_score = rouge.compute(
            predictions=[predictions[i]],
            references=[references[i]],
            use_stemmer=True
        )
        org_rouge1_score_list.append(rouge_score["rouge1"])
        org_rouge2_score_list.append(rouge_score["rouge2"])
        org_rougeL_score_list.append(rouge_score["rougeL"])
        org_rougeLsum_score_list.append(rouge_score["rougeLsum"])
    return org_rouge1_score_list, org_rouge2_score_list, org_rougeL_score_list, org_rougeLsum_score_list

def print_metrics_list(metric_name, metrics_list):
    print(f"{metric_name} length: {len(metrics_list)}; mean: {np.mean(metrics_list)*100:.2f}; [min, 25th, 50th, 75th, max]: [{min(metrics_list)*100:.2f}, {np.percentile(metrics_list, 25)*100:.2f}, {np.percentile(metrics_list, 50)*100:.2f}, {np.percentile(metrics_list, 75)*100:.2f}, {max(metrics_list)*100:.2f}]")