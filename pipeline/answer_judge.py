"""Answer matching with an LLM judge, as in the OpenForecaster authors' local_judge/llm_judge.py.

The prompt is their get_judge_prompt_with_gt (cot=False) verbatim. They judge with Llama-4-Scout; we use
Qwen3-8B (thinking off, greedy), the strongest open model on the NAS that is not one of the systems compared.
Used to drop generated distractors that restate the true answer (pipeline/build_openforesight_options.py) and to
grade OpenForecaster's free-form answers (baselines/baseline_openforecaster_freeform.py).
Run in the vllm venv (vLLM 0.11.0).
"""
import re

from vllm import SamplingParams

JUDGE = "Qwen/Qwen3-8B"


def judge_prompt(question, target, response):
    return f"""Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
For a response to "match", it must have the same information as in the ground-truth (not less nor unnecessary extra).
The response can be more specific than the ground-truth (for example, "Labrador" is more specific than "dog"), or have additional possible correct answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased.
For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be <= 1% for the response to be judged as a correct match. Here, if the ground truth is a specific numeric quantity but the response is a range, then they don't match (even if the range contains the ground truth).

Possible judgments:

"0": The response does not match the ground-truth answer.
"1": The response matches the ground-truth.

Question: "{question}"
Ground truth: "{target}"
Response: "{response}"

Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response. This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags.
YOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS."""


def matches(llm, tok, triples):
    """For each (question, ground truth, response), True if the judge says the response matches."""
    prompts = [tok.apply_chat_template([{"role": "user", "content": judge_prompt(*t)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for t in triples]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256))
    return [bool(re.search(r"<answer>\s*1\s*</answer>", o.outputs[0].text)) for o in outs]
