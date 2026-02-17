import torch
import torch.nn.functional as F
from transformers import Trainer

class KDTrainer(Trainer):
    """
    Knowledge Distillation 을 구현하기 위한 클래스
    loss 계산 역할을 수행
    """
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # teacher model 은 평가모드
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = input["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**model_inputs)
        with torch.no_grad():
            teacher_logits = self.teacher(**model_inputs).logits

        student_logits = student_logits[:, :-1, :].contiguous()
        teacher_logits = teacher_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, :-1, :].contiguous()

        # CELoss 계산
        ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # soft loss 계산
        log_p = F.log_softmax(student_logits / self.temperature, dim=-1)
        q = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_tok = F.kl_div(log_p, q, reduction="none").sum(-1)

        mask = (shift_labels != -100).float()
        kd = (kl_tok * mask).sum() / mask.sum().clamp_min(1.0)

        loss = (1 - self.alpha) * ce + self.alpha * (kd * (self.temperature * self.temperature))
        return (loss, outputs) if return_outputs else loss
    

def build_kd_features(tokenizer, example, max_len: int):
    """
    데이터셋의 conversations 에서 마지막 assistant 응답만 loss 에 반영하도록 하는 함수
    프롬프트 부분은 loss 에 반영하지 않게 하려는 의도
    """
    messages = example["conversations"]
    # 마지막 assistant
    last_a = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_a = i
            break
    if last_a is None:
        return None
    
    prompt_msgs = messages[:last_a]
    answer = messages[last_a]["content"]

    # "assistant + 빈 content" 까지 prompt
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs + [{"role": "assistant", "content": ""}],
        tokenize=False,
        add_generation_prompt=False,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_msgs + [{"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    m = min(len(prompt_ids), len(input_ids))
    labels = [-100] * m + input_ids[m:]
    labels = labels[: len(input_ids)]

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}   