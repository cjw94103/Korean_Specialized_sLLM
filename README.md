# 1. Introduction
<p align="center"><img src="https://github.com/cjw94103/KOSITP/assets/45551860/c645a919-5c60-4392-ad3d-9ad4150afa69" width="35%" height="35%"></p>
SLLM은 Smaller Large Language Model의 약자로 큰 언어 모델(Large Language Model) 중에서도 상대적으로 작은 크기를 가진 모델을 의미합니다. 이들은 흔히 말하는 SLM(Small Language Model)보다는 크지만, 최대 규모의 언어 모델에 비해서는 작습니다. 이 모델들은 여전히 대규모 데이터셋을 사용하여 학습되며, 복잡한 언어 이해 및 생성 작업을 수행할 수 있는 능력을 갖추고 있습니다. 기존의 좋은 성능의 언어 모델들을 그 크기가 매우 거대하여 (예를 들어, GPT 3.5의 경우 175B) 개인이 언어 모델을 학습하기 쉽지 않습니다. 이러한 이유로 본 프로젝트는 오픈 소스로 공개되어 있는 1 ~ 13B 사이의 pretrained model을 이용하고 AIHub, Kisti 등 다양한 한국어 데이터셋을 Instruction format으로 변환하여 한국어 대상의 Instruction Tuned 모델을 개발하고 자연스러운 출력을 위하여 DPO 등 Human Preference Learning을 구현하여 좋은 출력의 SLLM을 만드는 것을 목표로 합니다. 학습 프레임워크는 메모리 절약, 학습 속도 가속화를 위한 Unsloth Open Source Library를 사용하며 학습된 모델을 VLLM에서 사용할 수 있게 코드로 공개할 예정입니다. 또한 FastAPI를 통해 모델의 추론을 통신하고 Chainlit으로 간단한 홈페이지를 구현하여 웹 상에서의 챗봇을 구현해볼 예정입니다. 업데이트는 비주기적으로 될 예정입니다.

# 2. Update History
- 2024.11.09 : LLaMA_3.1_8B 사전학습 가중치를 이용한 Instruction SFT 학습 완료 : Enkeeper/LLaMA3.1_TaskInstruct_LoRA_SFT
- 2024.11.09 : LLaMA_3.1_8B 사전학습 가중치를 이용한 DPO 학습 완료 : Enkeeper/LLaMA3.1_TaskInstruct_LoRA_DPO

# 3. Dataset
데이터셋은 AIHub, Kisti에서 제공한 데이터셋을 사용하며 Instruction Tuning을 위하여 SuperNI(https://github.com/allenai/natural-instructions) 에 정의된 Task를 참고하여 가능한 23개의 Task Dataset으로 Reformatting을 진행하였습니다. 데이터셋 공개의 제한이 있어 sample_data 폴더 안에 Task 별 예제 데이터를 업로드하였습니다. Initial Dataset은 AIHUB을 사용하였으며 GPT-4o를 이용하여 정의된 Task대로 데이터를 생성하였습니다. 학습에 사용한 데이터셋의 총 개수는 31,260개 입니다. 
아래의 표는 각 Task에 대한 설명입니다.

|Task Name|Task IDX|Description|
|------|-|-------|
|Summarization - bullet type|task01_01|컨텍스트가 주어지면 컨텍스트 길이의 1/3 이상의 계층적 구조를 가진 요약 생성|
|Summarization – sentence type|task01_02|컨텍스트가 주어지면 컨텍스트 길이의 1/3 이상의 완전한 문장으로 요약 생성|
|Title Generation|task02|컨텍스트가 주어지면 컨텍스트를 대표하는 제목을 생성|
|Abstractive QA Objective Explanation - bullet type|task03_01|질문이 주어지면 객관식 문제의 형태로 보기, 정답, 정답에 대한 근거를 계층적 문장으로 생성|
|Abstractive QA Objective Explanation - sentence type|task03_02|질문이 주어지면 객관식 문제의 형태로 보기, 정답, 정답에 대한 근거를 완전한 문장으로 생성|
|Abstractive QA Subjective - bullet type|task04_01|질문이 주어지면 주관식 문제의 형태로 알맞은 정답을 계층적 문장으로 생성|
|Abstractive QA Subjective - sentence type|task04_02|질문이 주어지면 주관식 문제의 형태로 알맞은 정답을 완전한 문장으로 생성|
|Abstractive QA Yes or No Explanation - bullet type|task05_01|질문이 주어지면 예 또는 아니오로 정답을 생성하고 정답에 대한 근거를 계층적 문장으로 생성|
|Abstractive QA Yes or No Explanation - sentence type|task05_02|질문이 주어지면 예 또는 아니오로 정답을 생성하고 정답에 대한 근거를 완전한 문장으로 생성|
|Extractive QA Objective Explanation - bullet type|task06_01|컨텍스트가 주어지면 컨텍스트의 내용의 범위에서 객관식 문제의 형태로 질문, 보기, 정답, 정답에 대한 근거를 계층적 문장으로 생성|
|Extractive QA Objective Explanation - sentence type|task06_02|컨텍스트가 주어지면 컨텍스트의 내용의 범위에서 객관식 문제의 형태로 질문, 보기, 정답, 정답에 대한 근거를 완전한 문장으로 생성|
|Extractive QA Yes or No Explanation - bullet type|task07_02|컨텍스트가 주어지면 컨텍스트의 내용의 범위에서 질문, 보기, 정답 (예 또는 아니오), 정답에 대한 근거를 계층적 문장으로 생성|
|Extractive QA Yes or No Explanation - sentence type|task07_02|컨텍스트가 주어지면 컨텍스트의 내용의 범위에서 질문, 보기, 정답 (예 또는 아니오), 정답에 대한 근거를 완전한 문장으로 생성|
|Extractive QA Subjective - bullet type|task08_01|컨텍스트가 주어지면 주관식 문제의 형태로 질문, 정답을 계층적 문장으로 생성|
|Extractive QA Subjective - senrence type|task08_02|컨텍스트가 주어지면 주관식 문제의 형태로 질문, 정답을 완전한 문장으로 생성|
|Text Completion|task09|미완성 형태의 컨텍스트가 주어지면 컨텍스트의 내용의 범위에서 나머지 부분을 생성|
|Title2Contents Generation - bullet type|task10_01|짧은 제목이 주어지면 주어진 제목과 관련된 텍스트를 계층적 문장으로 생성|
|Title2Contents Generation - sentence type|task10_02|짧은 제목이 주어지면 주어진 제목과 관련된 텍스트를 완전한 문장으로 생성|
|Keyword Tagging|task11|컨텍스트가 주어지면 컨텍스트의 내용을 대표하는 다수의 핵심 키워드를 생성|
|Table QA - bullet type|task12_01|HTML 형태의 표 컨텍스트가 주어지면 질문, 정답을 계층적 문장으로 생성|
|Table QA - sentence type|task12_02|HTML 형태의 표 컨텍스트가 주어지면 질문, 정답을 완전한 문장으로 생성|
|Paraphrasing - bullet type|task13_01|컨텍스트가 주어지면 컨텍스트의 내용의 의미가 왜곡되지 않게 다른 표현으로 계층적 문장 생성|
|Paraphrasing - sentence type|task13_02|컨텍스트가 주어지면 컨텍스트의 내용의 의미가 왜곡되지 않게 다른 표현으로 완전한 문장 생성|



# 4. Framework
학습은 unsloth를 기반으로 수행됩니다. https://github.com/unslothai/unsloth 링크를 참고하여 unsloth 라이브러리를 설치하여주세요. 나머지 dependency는 requirements.txt를 참고하여 설치하여 주시기 바랍니다.

# 5. Instruction Supervised Fine Tuning
학습 구현은 unsloth 프레임워크 (https://github.com/unslothai/unsloth) 을 기반으로 합니다. unsloth는 LoRA fine-tuning에 대하여 빠른 학습 속도와 좋은 GPU 메모리 효율을 보여줍니다. 또한 다양한 Open Foundation 모델을 학습할 수 있습니다. 따라서 이 구현에서의 모든 모델은
Unsloth를 이용한 LoRA Fine-Tuning을 수행합니다.

- 아래와 같은 명령어를 사용하여 Supervised Fine Tuning을 수행합니다. args에 대한 자세한 내용은 train_LoRA_SFT.py를 참고해주세요.
```python
$ python train_LoRA_SFT.py --[args]
```

# 6. Directly Preference Optimization
DPO 역시 unsloth 프레임워크 (https://github.com/unslothai/unsloth) 을 기반으로 합니다.

- 아래와 같은 명령어를 사용하여 Directly Preference Optimization을 수행합니다. args에 대한 자세한 내용은 train_LoRA_DPO.py를 참고해주세요.
```python
$ python train_LoRA_DPO.py --[args]
```

# 5. Inference
LoRA Fine-Tuning이 완료된 모델은 추론을 수행할 수 있습니다. 학습이 완료된 후 저장된 weight의 폴더 또는 huggingface에 업로드된 모델을 로드할 수 있습니다. 자세한 내용은 inference.py를 참고해주세요. llama3.1 추론에 대한 샘플 코드 및 추론 결과는 아래와 같습니다.
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name , 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map='auto'
    # token = "hf_...", 
)
system_prompt = """당신은 경영 컨설턴트 비서 인공지능입니다. 친절하게 설명해주세요. 답이 어렵다면 단계별로 설명해주세요.\n\n"""
instruction = "ESG 경영이란 무엇인가?"

system_content = """질문에 대한 올바른 답변을 상세하고 면밀하게 작성해 주세요. 아래와 같은 규칙을 지켜주세요.
1. 답변은 심층적 분석을 포함하며, 전문적인 어조를 사용해야합니다.
2. 답변은 질문에 대한 깊이 있는 설명과 예시를 제공해야 합니다.
3. 답변은 bullet points를 사용해 독자가 쉽게 이해할 수 있는 명확한 계층 구조를 갖춰야 합니다. 
4. 모든 답변은 Markdown format으로 작성되어야 하며, 명시적으로 ```markdown ```을 사용하지 않도록 주의해서 작성하세요."""
user_content = '###Question###:\n양자 컴퓨팅이란 무엇인가?\n\n###Answer###:'

input_message = make_chat_dict(system_content, user_content)
input_message = tokenizer.apply_chat_template(input_message, tokenize=False, add_generation_prompt=True)
input_tokens = tokenizer(input_message, return_tensors='pt', return_token_type_ids=False)

result = model.generate(**input_tokens,
                        max_new_tokens=4096, early_stopping=True, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, 
                        temperature=0.3, streamer=streamer)

"""
**개요 및 정의**

- 양자 컴퓨팅은 기존 전통적인 컴퓨팅 방식과는 다른 새로운 기술 분야입니다.
- 이 기술은 양자역학의 원리를 활용하여 정보 처리 및 계산 능력을 극대화하려는 목표를 가지고 있습니다.
- 양자 컴퓨팅의 핵심은 양자 비트(qubit)를 사용하는 것입니다.

**양자 비트(qubit)의 특징**

- 기존 비트는 두 가지 상태(0과 1)만을 가질 수 있지만, qubit은 여러 상태를 동시에 유지할 수 있습니다.
- 이로 인해 양자 컴퓨팅에서는 여러 가능한 해를 동시에 탐색할 수 있습니다.
- 이러한 특성은 특히 복잡한 문제 해결에 유리하게 작용합니다.

**양자 컴퓨팅의 주요 이점**

- **고속 계산 능력**: 
  - 양자 컴퓨팅은 특정 문제에서 기존 컴퓨터보다 훨씬 빠른 계산을 가능하게 합니다.
  - 특히, 대규모 데이터 셋 및 복잡한 수학적 연산에 유리합니다.

...
...

"""
```

자세한 내용은 inference.py를 참고해주세요.

# 7. vLLM를 이용한 추론
unsloth는 vLLM에서의 모델 추론을 위한 메서드를 지원합니다. 아래와 같은 코드를 이용하여 모델을 저장합니다.
```python
model.save_pretrained_merged('weights_directory_path', tokenizer, save_method="merged_16bit")
```
위 코드는 lora_adapter와 모델을 통합하여 저장한 후 vLLM에서 추론할 수 있게 해줍니다. lora adapter만 저장을 원하는 경우 아래와 같은 코드를 사용해주세요.
```python
model.save_pretrained_merged('weights_directory_path', tokenizer, save_method="lora")
```

vLLM을 이용한 offline inference에 대한 예제는 https://docs.vllm.ai/en/v0.5.5/getting_started/examples/offline_inference.html 를 참고해주세요.

# 8. Custom 모델 기반의 챗봇 구축
별도 저장소에 구현 예정
