# 1. Introduction
<p align="center"><img src="https://github.com/cjw94103/KOSITP/assets/45551860/c645a919-5c60-4392-ad3d-9ad4150afa69" width="35%" height="35%"></p>
SLLM은 Smaller Large Language Model의 약자로 큰 언어 모델(Large Language Model) 중에서도 상대적으로 작은 크기를 가진 모델을 의미합니다. 이들은 흔히 말하는 SLM(Small Language Model)보다는 크지만, 최대 규모의 언어 모델에 비해서는 작습니다. 이 모델들은 여전히 대규모 데이터셋을 사용하여 학습되며, 복잡한 언어 이해 및 생성 작업을 수행할 수 있는 능력을 갖추고 있습니다. 기존의 좋은 성능의 언어 모델들을 그 크기가 매우 거대하여 (예를 들어, GPT 3.5의 경우 175B) 개인이 언어 모델을 학습하기 쉽지 않습니다. 이러한 이유로 본 프로젝트는 오픈 소스로 공개되어 있는 1 ~ 13B 사이의 pretrained model을 이용하고 AIHub, Kisti 등 다양한 한국어 데이터셋을 Instruction format으로 변환하여 한국어 대상의 Instruction Tuned 모델을 개발하고 자연스러운 출력을 위하여 DPO 등 강화학습 방법을 구현하여 좋은 출력의 SLLM을 만드는 것을 목표로 합니다. 학습 프레임워크는 메모리 절약, 학습 속도 가속화를 위한 Unsloth Open Source Library를 사용하며 학습된 모델을 VLLM에서 사용할 수 있게 코드로 공개할 예정입니다. 또한 FastAPI를 통해 모델의 추론을 통신하고 Chainlit으로 간단한 홈페이지를 구현하여 웹 상에서의 챗봇을 구현해볼 예정입니다. 업데이트는 비주기적으로 될 예정입니다.

# 2. Update History
- 2024.11.09 : LLaMA_3.1_8B 사전학습 가중치를 이용한 Instruction SFT 학습 완료 : Enkeeper/LLaMA3.1_TaskInstruct_LoRA_SFT
- 2024.06.20 : LLaMA_3.1_8B 사전학습 가중치를 이용한 DPO 학습 완료 : Enkeeper/LLaMA3.1_TaskInstruct_LoRA_DPO

# 3. Dataset
데이터셋은 AIHub, Kisti에서 제공한 데이터셋을 사용하며 Instruction Tuning을 위하여 SuperNI(https://github.com/allenai/natural-instructions) 에 정의된 Task를 참고하여 가능한 23개의 Task Dataset으로 Reformatting을 진행하였습니다. 데이터셋 공개의 제한이 있어 sample_data 폴더 안에 Task 별 예제 데이터를 업로드하였습니다. Initial Dataset은 AIHUB을 사용하였으며 GPT-4o를 이용하여 정의된 Task대로 데이터를 생성하였습니다. 학습에 사용한 데이터셋의 총 개수는 31,260개 입니다. 
아래의 표는 각 Task에 대한 설명입니다.

|Task Name|Description|
|------|-------|
|Summarization - bullet type|컨텍스트가 주어지면 컨텍스트 길이의 1/3 이상의 계층적 구조를 가진 요약 생성|
|Summarization – sentence type|컨텍스트가 주어지면 컨텍스트 길이의 1/3 이상의 완전한 문장으로 요약 생성|
|Title Generation|컨텍스트가 주어지면 컨텍스트를 대표하는 제목을 생성|
|Abstractive QA Objective Explanation - bullet type|질문이 주어지면 객관식 문제의 형태로 보기, 정답, 정답에 대한 근거를 계층적 문장으로 생성|
|Abstractive QA Objective Explanation - sentence type|질문이 주어지면 객관식 문제의 형태로 보기, 정답, 정답에 대한 근거를 완전한 문장으로 생성|
|Abstractive QA Subjective - bullet type|질문이 주어지면 주관식 문제의 형태로 알맞은 정답을 계층적 문장으로 생성|
|Abstractive QA Subjective - sentence type|질문이 주어지면 주관식 문제의 형태로 알맞은 정답을 완전한 문장으로 생성|
|Abstractive QA Yes or No Explanation - bullet type|질문이 주어지면 예 또는 아니오로 정답을 생성하고 정답에 대한 근거를 계층적 문장으로 생성|
|Abstractive QA Yes or No Explanation - sentence type|질문이 주어지면 예 또는 아니오로 정답을 생성하고 정답에 대한 근거를 완전한 문장으로 생성|


# 4. Framework
학습은 unsloth를 기반으로 수행됩니다. https://github.com/unslothai/unsloth 링크를 참고하여 unsloth 라이브러리를 설치하여주세요. 나머지 dependency는 requirements.txt를 참고하여 설치하여 주시기 바랍니다.

# 4. Train
먼저 config.json 파일을 만들어야 합니다. make_train_config.ipynb와 config 폴더 안에 있는 예시 config 파일을 참고하여 config 파일을 만들어주세요. 학습 구현은 unsloth 프레임워크 (https://github.com/unslothai/unsloth) 을 기반으로 합니다. unsloth는 LoRA fine-tuning에 대하여 빠른 학습 속도와 좋은 GPU 메모리 효율을 보여줍니다. 또한 다양한 Open Foundation 모델을 학습할 수 있습니다. 따라서 이 구현에서의 모든 모델은 LoRA Fine-Tuning을 수행합니다. 학습 또는 추론에 사용 할 특정 GPU의 선택을 원하지 않는 경우 코드에서 os.environ["CUDA_VISIBLE_DEVICES"]="1"를 주석처리 해주세요.
- config 파일을 만들고 아래와 같은 명령어를 사용하여 LoRA Fine-Tuning을 수행합니다.
```python
$ python train_LoRA.py --config_path /path/your/config_path
```

# 5. Inference
LoRA Fine-Tuning이 완료된 모델은 추론을 수행할 수 있습니다. 학습이 완료된 후 저장된 weight의 폴더 또는 huggingface에 업로드된 모델을 로드할 수 있습니다. 자세한 내용은 inference.ipynb 파일을 참고하여주세요. 추론에 대한 샘플 코드 및 추론 결과는 아래와 같습니다.
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

input_text = preprocess_text(system_prompt + instruction)
input_tokens = tokenizer(input_text, return_tensors='pt', return_token_type_ids=False)

result = model.generate(**input_tokens,
                        max_new_tokens=1024, early_stopping=True, do_sample=False, eos_token_id=2, pad_token_id=2, temperature=0.7,
                        streamer=streamer)

"""
ESG 경영은 기업이 환경, 사회, 지배 구조에 대한 영향력을 고려하여 경영하는 접근 방식입니다. 이는 기업이 사회적 책임을 다하고 지속 가능한 비즈니스 모델을 구축하는 것을 강조합니다. 

"E"는 환경을 의미하며, 기업이 환경에 미치는 영향을 고려하고 친환경 방식으로 운영하는 것을 의미합니다. 이는 기업이 탄소 배출을 줄이고 재활용을 촉진하며 지속 가능한 에너지 소스를 사용하는 등의 방법으로 환경에 대한 영향을 최소화하는 것을 의미합니다.
"S"는 사회를 의미하며, 기업이 사회에 미치는 영향을 고려하고 사회적 책임을 다하는 것을 의미합니다. 이는 기업이 다양성과 포용성을 증진하고 공정한 노동 조건을 제공하며 지역 사회에 기여하는 등의 방법으로 사회에 긍정적인 영향을 미치는 것을 의미합니다.
"G"는 지배 구조를 의미하며, 기업이 지배 구조에 대한 영향력을 고려하고 투명하고 윤리적인 방식으로 운영하는 것을 의미합니다. 이는 기업이 독립적인 이사회를 구성하고 공정한 거래를 하며 공정한 보상 체계를 구축하는 등의 방법으로 지배 구조에 대한 영향력을 최소화하는 것을 의미합니다.

ESG 경영은 기업이 사회적 책임을 다하고 지속 가능한 비즈니스 모델을 구축하는 것을 강조하는 접근 방식입니다. 이는 기업이 환경, 사회, 지배 구조에 대한 영향력을 고려하여 경영하는 것을 의미합니다.$&%</s>
"""
```
# 6. vLLM를 이용한 추론
unsloth는 vLLM에서의 모델 추론을 위한 메서드를 지원합니다. 아래와 같은 코드를 이용하여 모델을 저장합니다.
```python
model.save_pretrained_merged('folder_path', tokenizer, save_method="merged_16bit")
```
위 코드는 lora_adapter와 모델을 통합하여 저장한 후 vLLM에서 추론할 수 있게 해줍니다. lora adapter만 저장을 원하는 경우 아래와 같은 코드를 사용해주세요.
```python
model.save_pretrained_merged('folder_path', tokenizer, save_method="lora")
```
보다 자세한 내용은 inference.ipynb와 vLLM_offline_inference.ipynb 파일을 참고해주세요

# 7. Zero-Shot Performance
### Zero-Shot이란?
Zero-Shot은 쉽게 말하면 "모델이 학습 과정에서 배우지 않은 작업을 수행하는 것"을 의미합니다. sLLM 기반의 Instruction Tuned 모델은 학습 과정에서 배우지 않는 태스크에 대한 설명을 제공하면 태스크를 자연어를 통해 이해하여 Zero-Shot Inference 일정 수준 충족할 수 있습니다.
### Evaluation Dataset
모델의 Zero-Shot 능력을 정량적으로 평가하기 위해 분류 태스크를 활용합니다. 4개의 데이터셋을 사용하여 평가를 진행하며 이 데이터셋은 학습 과정에서 전혀 사용하지 않은 데이터셋입니다. 각 모델별 데이터셋에 대한 F1-Score로 모델 성능을 비교합니다.
Zero-Shot Dataset의 설명은 아래와 같습니다.
- KorNLI : 한국어로 구성된 Natural Language Inference 태스크로써 클래스는 entailment, neutral, contradiction으로 표시됩니다. (https://github.com/kakaobrain/kor-nlu-datasets)
- KorQuestionPair : 한국어로 구성된 두 개의 문장에 대하여 유사성을 평가하는 태스크로써 클래스는 두 문장이 같을 경우 0, 같지 않을 경우 1로 표시됩니다. (https://github.com/songys/Question_pair)
- KoreaHateSpeech : 한국어로 구성된 사회적 편견 및 혐오 표현을 탐지하는 태스크로써 여기서는 사회적 편견의 유무에 대한 태스크를 수행합니다. 사회적 편견이 있는 경우 1, 없는 경우 0으로 표시됩니다. (https://github.com/songys/Question_pair)
- NSMC : 한국어로 구성된 영화 리뷰 데이터셋이며 긍정, 부정의 레이블이 존재합니다. (https://github.com/e9t/nsmc)

모델에게 태스크 이해를 위한 시스템 프롬프트는 최소한으로 입력하여 각 모델별, 데이터셋별 Average F1-Score를 산출합니다. 성능은 아래와 같습니다.
|모델|KorNLI|KorQuestionPair|KoreaHateSpeech|NSMC|
|------|---|---|---|---|
|SOLAR 10.7B TaskInstruct|0.44|0.68|0.63|0.76|
|Mistral 7B TaskInstruct|0.31|0.51|0.41|0.77|
|Gemma 2B TaskInstruct|0.30|0.45|0.49|0.65|
