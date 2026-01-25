from vllm import LLM, SamplingParams

prompts = [
    [{"role": "user", "content": "Explain how wonderful you are"}],
    [{"role": "user", "content": "너가 얼마나 대단한 지 설명해 봐"}],
]

sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)

llm = LLM(model="EXAONE-4.0-1.2B-GPTQ")
outputs = llm.chat(prompts, sampling_params)

for output in outputs:
    print("#################")
    print(output.outputs[0].text)
    print()