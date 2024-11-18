# 1. Introduction
<p align="center"><img src="https://github.com/cjw94103/KOSITP/assets/45551860/c645a919-5c60-4392-ad3d-9ad4150afa69" width="35%" height="35%"></p>
SLLM은 Smaller Large Language Model의 약자로 큰 언어 모델(Large Language Model) 중에서도 상대적으로 작은 크기를 가진 모델을 의미합니다. 이들은 흔히 말하는 SLM(Small Language Model)보다는 크지만, 최대 규모의 언어 모델에 비해서는 작습니다. 이 모델들은 여전히 대규모 데이터셋을 사용하여 학습되며, 복잡한 언어 이해 및 생성 작업을 수행할 수 있는 능력을 갖추고 있습니다. 기존의 좋은 성능의 언어 모델들을 그 크기가 매우 거대하여 (예를 들어, GPT 3.5의 경우 175B) 개인이 언어 모델을 학습하기 쉽지 않습니다. 이러한 이유로 본 프로젝트는 오픈 소스로 공개되어 있는 1 ~ 13B 사이의 pretrained model을 이용하고 AIHub, Kisti 등 다양한 한국어 데이터셋을 Instruction format으로 변환하여 한국어 대상의 Instruction Tuned 모델을 개발하고 자연스러운 출력을 위하여 DPO 등 Human Preference Learning을 구현하여 좋은 출력의 SLLM을 만드는 것을 목표로 합니다. 학습 프레임워크는 메모리 절약, 학습 속도 가속화를 위한 Unsloth Open Source Library를 사용하며 학습된 모델을 VLLM에서 사용할 수 있게 코드로 공개할 예정입니다. 또한 FastAPI를 통해 모델의 추론을 통신하고 Chainlit으로 간단한 홈페이지를 구현하여 웹 상에서의 챗봇을 구현해볼 예정입니다. 업데이트는 비주기적으로 될 예정입니다. 코드를 올바르게 실행하기 위하여 아래와 같은 논문을 읽어볼것을 추천드립니다.

- Instruction Tuning에 대한 Survey 논문 : https://arxiv.org/abs/2308.10792
- LoRA Fine-Tuning에 관한 논문 : https://arxiv.org/abs/2106.09685
- Directly Preference Optimization에 관한 논문 : https://arxiv.org/abs/2305.18290

# 2. Update History
- 2024.11.09 : LLaMA_3.1_8B 사전학습 가중치를 이용한 Instruction SFT 학습 완료 : Enkeeper/LLaMA3.1_TaskInstruct_LoRA_SFT
- 2024.11.09 : LLaMA_3.1_8B DPO 학습 완료 : Enkeeper/LLaMA3.1_TaskInstruct_LoRA_DPO
- 2024.11.18 : SoLAR_10.7B 사전학습 가중치를 이용한 Instruction SFT 학습 완료 : Enkeeper/SoLAR_TaskInstruct_LoRA_SFT
- 2024.11.18 : SoLAR_10.7B DPO 학습 완료 : Enkeeper/SoLAR_TaskInstruct_LoRA_DPO

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

# 7. Inference
이 저장소에서는 vLLM을 이용한 추론 파이프라인을 지원합니다. 추론은 아래와 같은 과정으로 수행됩니다.

- unsloth를 이용하여 학습한 모델의 LoRA weights를 로드 후 merged_16 bits로 저장
- vLLM를 통한 Offline Inference 수행

### merged 16 bits 저장
먼저, unsloth를 이용하여 학습한 모델의 LoRA weights를 로드하여 merged_16 bits로 저장합니다. 이 과정은 모두 unsloth framework 내에서 수행할 수 있습니다. 자세한 내용은 merged_weights.py를 참고해주세요. 주요 코드는 아래와 같습니다.
```python
model.save_pretrained_merged(args.merged_weights_path, tokenizer, save_method="merged_16bit")
```

### vLLM을 이용한 추론
merged 16 bits 저장된 weights를 로드하여 추론을 수행합니다. 자세한 내용은 vllm_inference.py를 참고해주세요. 주요 코드는 아래와 같습니다.
```python
llm = LLM(model=args.model_id, 
          tensor_parallel_size=args.tensor_parallel_size, 
          gpu_memory_utilization=args.gpu_memory_utilization)

sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, presence_penalty=args.presence_penalty,
                                frequency_penalty=args.frequency_penalty, repetition_penalty=args.repetition_penalty, top_p=args.top_p,
                                top_k=args.top_k, min_p=args.min_p, use_beam_search=args.use_beam_search, length_penalty=args.length_penalty)

outputs = llm.generate(input_text_list, sampling_params)
```


자세한 내용은 inference.py를 참고해주세요.


# 8. Evaluation
구현 예정
