---
title: "[AI 기초 15] PyTorch 실전 도구 모음: argparse, 모델 저장·불러오기, TensorBoard, Custom Transform, Knowledge Distillation"
date: 2026-07-02 22:18:00 +0900
categories: [Neural Network, Basics]
tags: [pytorch, argparse, state-dict, checkpoint, tensorboard, data-augmentation, custom-transform, knowledge-distillation]
description: PyTorch로 실험할 때 매번 쓰는 도구들 — argparse, state_dict와 체크포인트 저장·불러오기, TensorBoard, Custom Transform, Knowledge Distillation — 을 흔한 실수와 함께 코드로 정리합니다.
math: true
---

> 시리즈의 마지막 글입니다. 논문 코드를 돌려 보거나 직접 실험을 할 때 **매번 쓰게 되는 도구들**을 한곳에 모았습니다. 실행 옵션 받기(argparse), 모델 저장과 불러오기, 학습 곡선 기록(TensorBoard), 직접 만든 Data Augmentation, 그리고 큰 모델의 지식을 작은 모델로 옮기는 Knowledge Distillation까지 다룹니다.

## 1. argparse: 실행할 때 옵션 주기

### 왜 필요한가

batch size, epoch, learning rate 같은 값을 바꿀 때마다 코드를 열어 고치면 번거롭고 실수도 생깁니다. **argparse**는 파이썬 표준 라이브러리의 명령행 인자 처리 모듈로, 실행할 때 옵션을 넘길 수 있게 해 줍니다. ReActNet 같은 논문 공식 코드도 대부분 이 방식을 씁니다.

### 사용 순서

1. `argparse` import
2. Parser 객체 생성
3. 처리할 인자 추가 (`add_argument`)
4. `parse_args()` 실행

```python
# train.py
import argparse

parser = argparse.ArgumentParser(description="CIFAR-10 training")
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--epochs", type=int, default=256, help="num of training epochs")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="init learning rate")
parser.add_argument("--save", type=str, default="./models", help="checkpoint dir")
parser.add_argument("--teacher", type=str, default="resnet34", help="teacher model name")
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
args = parser.parse_args()

print(args.batch_size, args.learning_rate)
print(vars(args))        # Namespace → dict 로 바꿔 한 번에 로그로 남기기 좋음
```

```bash
python train.py --batch_size 128 --learning_rate 0.01 --resume
```

`vars(args)`를 쓰면 객체의 변수를 dictionary 형태로 바꿀 수 있어, 실험 설정을 로그나 체크포인트에 같이 저장할 때 편합니다.

## 2. 모델 저장하기와 불러오기

PyTorch는 Python의 **pickle**로 객체를 바이너리로 **직렬화(serialize)**해 저장하고, 불러올 때 **역직렬화(deserialize)**해서 메모리에 올립니다. 저장 시점은 보통 학습이 끝난 후, 또는 특정 epoch마다입니다.

### 방법 1. 모델 전체 저장

```python
torch.save(model, "model.pt")              # 구조 + 파라미터
model = torch.load("model.pt", weights_only=False)
model.eval()
```

간단하지만, 저장할 때의 **클래스 정의와 디렉터리 구조에 묶여** 코드가 바뀌면 불러오지 못할 수 있습니다.

> PyTorch 2.6부터 `torch.load`의 기본값이 `weights_only=True`로 바뀌어, 텐서·숫자·dict 같은 기본 자료형만 불러옵니다. 모델 객체 전체를 불러오려면 위처럼 `weights_only=False`를 줘야 하는데, 이 경우 pickle이 **파일 안의 임의 코드를 실행할 수 있으므로** 출처를 믿을 수 있는 파일에만 써야 합니다. 이것도 state_dict 저장을 권장하는 이유입니다.
{: .prompt-warning }

### 방법 2. state_dict 저장 (권장)

**state_dict**는 각 층의 학습 가능한 파라미터(weight, bias)와 BN의 running mean/var 같은 버퍼를 **층 이름 → 텐서**로 매핑한 Python dictionary입니다. 추론에는 학습된 파라미터만 있으면 되므로, 이것만 저장하는 것이 가장 가볍고 유연합니다.

```python
print(model.state_dict().keys())
# odict_keys(['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', ...])

torch.save(model.state_dict(), "cifar_net.pt")

model = MyNetwork()                                    # 같은 구조를 먼저 만들고
model.load_state_dict(torch.load("cifar_net.pt"))      # 파라미터를 매핑
model.eval()
```

**불러올 모델은 저장한 모델과 같은 구조여야 합니다.** 구조가 아니라 파라미터만 저장했기 때문입니다.

### model.eval()을 잊지 말 것

`model.eval()`은 **학습에만 필요한 동작을 끕니다.**

- **Dropout**: 뉴런을 끄지 않음
- **BatchNorm**: 배치 통계 대신 저장된 running mean/var 사용

빠뜨리면 추론 결과가 매번 달라지거나 정확도가 이상하게 낮게 나옵니다. 기울기 계산도 필요 없으므로 `torch.no_grad()`와 함께 씁니다.

```python
model.eval()
with torch.no_grad():
    pred = model(x).argmax(dim=1)
```

### 방법 3. Checkpoint: 학습을 이어서 하기

학습을 중간에 멈췄다가 이어 하려면 모델 파라미터뿐 아니라 **optimizer 상태(momentum 등), scheduler 상태, epoch, 최고 성능**도 함께 저장해야 합니다. dictionary에 모두 넣어 저장합니다. 확장자는 관례적으로 `.tar` 또는 `.pth.tar`를 씁니다.

```python
import os, logging

def save_checkpoint(state, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, "checkpoint.pth.tar"))

# 저장
save_checkpoint({
    "epoch": epoch,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_top1_acc": best_top1_acc,
}, args.save)

# 불러오기
start_epoch, best_top1_acc = 0, 0.0
checkpoint_tar = os.path.join(args.save, "checkpoint.pth.tar")
if os.path.exists(checkpoint_tar):
    logging.info(f"loading checkpoint {checkpoint_tar}")
    checkpoint = torch.load(checkpoint_tar, map_location="cpu")
    start_epoch = checkpoint["epoch"] + 1
    best_top1_acc = checkpoint["best_top1_acc"]
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
```

- `os.path.join`: 디렉터리 경로와 파일 이름을 합쳐 줌
- `os.path.exists`: 체크포인트가 있는지 확인
- `logging.info`: 진행 상황 추적용 로그
- `strict=False`: **key가 일치하는 층만** 불러오고 나머지는 무시. 구조를 조금 바꾼 모델에 사전학습 가중치를 부분적으로 넣을 때 유용. 단, **학습 재개처럼 구조가 같아야 하는 상황에서는 `strict=True`(기본값)**를 쓰는 편이 안전합니다. 아래처럼 key가 하나도 맞지 않아도 에러 없이 넘어가기 때문입니다.
- `map_location`: GPU에서 저장한 것을 CPU에서 열 때 필요

```python
import torch
import torch.nn as nn

dp_model = nn.DataParallel(nn.Linear(2, 2))
print(list(dp_model.state_dict().keys()))            # ['module.weight', 'module.bias']

result = nn.Linear(2, 2).load_state_dict(dp_model.state_dict(), strict=False)
print(result)
# _IncompatibleKeys(missing_keys=['weight', 'bias'],
#                   unexpected_keys=['module.weight', 'module.bias'])
```

`DataParallel`로 감싼 모델을 그대로 저장하면 모든 key 앞에 `module.`이 붙습니다. 이걸 감싸지 않은 모델에 `strict=False`로 넣으면 **아무 가중치도 들어가지 않았는데 에러도 나지 않습니다.** 저장할 때 `model.module.state_dict()`로 저장하거나, 불러올 때 반환값의 `missing_keys`를 꼭 확인하세요.

### 사전학습 모델 불러오기와 고정하기

```python
import torchvision.models as models

model_teacher = models.__dict__[args.teacher](weights="DEFAULT")   # 이름으로 모델 생성 + 사전학습 가중치
model_teacher = torch.nn.DataParallel(model_teacher).cuda()

for p in model_teacher.parameters():
    p.requires_grad = False          # 기울기 계산 끔 → 상수 취급, 업데이트 안 됨
model_teacher.eval()
```

`nn.DataParallel`은 같은 모델을 여러 GPU에 복사하고, 입력 배치를 GPU별로 나눠 각자 forward/backward를 수행합니다. (최근에는 더 효율적인 `DistributedDataParallel`을 권장합니다.)

## 3. TensorBoard: 학습 곡선 기록하기

논문에 나오는 "epoch에 따른 accuracy·loss 그래프"를 그리려고 매 epoch 값을 txt로 저장했다가 다시 읽는 것은 번거롭습니다. **TensorBoard**를 쓰면 학습 중에 바로 기록하고 브라우저에서 볼 수 있습니다.

### 환경 확인 (Linux + Anaconda)

```bash
conda env list
```

```bash
conda activate myenv
```

```bash
conda list | grep tensorboard
```

없으면 설치합니다. `torch.utils.tensorboard`는 PyTorch 1.1에서 실험 기능으로 들어왔고, 1.2(2019년 8월)에서 experimental 표시가 빠지며 정식으로 지원됐습니다.

```bash
pip install tensorboard
```

### 기록하기

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/resnet18_lr0.01")      # 경로 생략 시 ./runs/날짜_호스트명

for epoch in range(epochs):
    train_loss = train_one_epoch(model)
    val_acc = evaluate(model)

    writer.add_scalar("Loss/train", train_loss, epoch)      # (tag, value, global_step)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    for name, param in model.named_parameters():            # 가중치 분포 히스토그램
        writer.add_histogram(name, param, epoch)

writer.add_graph(model, torch.randn(1, 3, 32, 32))          # 모델 구조
writer.flush()
writer.close()
```

- **tag**: 그래프 이름. `Loss/train`, `Loss/val`처럼 `/`로 묶으면 같은 그룹으로 표시
- **scalar_value**: 기록할 값
- **global_step**: x축 (epoch 또는 iteration)
- 기록이 끝나면 `flush()`로 디스크에 쓰고 `close()`

### 보기

```bash
tensorboard --logdir runs --port 6006
```

브라우저에서 `localhost:6006`을 열면 원래 값(RealValue)과 부드럽게 만든 값(SmoothedValue)이 함께 그려집니다. 여러 실험 폴더를 `runs/` 아래에 두면 **한 화면에서 비교**할 수 있어, 예를 들어 "Pruning 비율에 따른 accuracy" 같은 그래프를 쉽게 만들 수 있습니다. 원격 서버라면 SSH 포트 포워딩(`ssh -L 6006:localhost:6006 서버주소`)으로 볼 수 있습니다.

## 4. Custom Data Augmentation 만들기

[5편]({% post_url 2026-05-16-overfitting-regularization-augmentation %})에서 본 기본 augmentation은 `torchvision.transforms`에 대부분 있지만, 직접 만들어야 할 때도 있습니다.

### ToTensor와 Normalize

PyTorch는 텐서로 작업하므로, 이미지(PIL)나 배열(numpy)은 `ToTensor()`로 바꿔야 합니다. 이때 픽셀 값 0~255가 **0~1로** 바뀌고, 순서도 (H, W, C)에서 (C, H, W)로 바뀝니다.

`Normalize(mean, std)`는 채널마다 $$(x - \text{mean}) / \text{std}$$를 계산합니다.

- `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` → 0~1을 **-1~1**로
- `Normalize((0, 0, 0), (1, 1, 1))` → 원본 그대로 (augmentation 결과를 눈으로 확인할 때)
- 실제 학습에서는 **데이터셋의 채널별 평균과 표준편차**를 구해 넣습니다.

### Custom Transform 클래스

`__call__`을 구현하면 클래스를 **함수처럼 호출**할 수 있어 `transforms.Compose`에 그대로 넣을 수 있습니다. `__init__`에서 출력 크기 같은 설정을 받아 저장합니다.

```python
import numpy as np
from skimage import transform as sk_transform

class Rescale:
    """짧은 변을 output_size에 맞추고 비율 유지 (int), 또는 (h, w)로 정확히 맞춤 (tuple)"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):                       # image: numpy (H, W, C)
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        return sk_transform.resize(image, (int(new_h), int(new_w)))

class RandomCrop:
    def __init__(self, output_size):
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h + 1)    # 0 ~ h-new_h 중 무작위
        left = np.random.randint(0, w - new_w + 1)
        return image[top: top + new_h, left: left + new_w]   # 배열 슬라이싱으로 잘라내기
```

`image[행 시작:행 끝, 열 시작:열 끝]` 슬라이싱으로 특정 영역을 잘라냅니다. 끝 인덱스는 포함되지 않습니다.

> `skimage.transform.resize`는 결과를 **0~1 범위의 float64** 배열로 돌려줍니다. 이 배열에 `ToTensor()`를 쓰면 이미 0~1이라 255로 나누지 않고, dtype도 **`torch.float64`** 그대로 남습니다. 모델 가중치는 float32라서 그대로 넣으면 dtype 오류가 납니다. 아래처럼 확인용으로 그림만 그릴 때는 괜찮지만, 학습에 쓸 때는 `ToTensor()` 뒤에 `T.ConvertImageDtype(torch.float32)`를 붙이거나 `Lambda(lambda x: x.float())`로 바꿔 주세요.
{: .prompt-warning }

### Dataset에 적용하고 확인하기

```python
import torchvision
from torchvision import transforms as T
import matplotlib.pyplot as plt

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

transform = T.Compose([
    T.Lambda(lambda img: np.asarray(img)),     # PIL → numpy (skimage용)
    Rescale(40),
    RandomCrop(32),
    T.ToTensor(),
    T.Normalize((0, 0, 0), (1, 1, 1)),         # 확인용: 원본 그대로
])

trainset = torchvision.datasets.CIFAR10(
    root="./data",       # 저장 위치
    train=True,          # train set이면 True, test set이면 False
    download=True,       # 이미 받았다면 False
    transform=transform,
)

img, label = trainset[0]
plt.imshow(img.permute(1, 2, 0).numpy())
plt.title(classes[label]); plt.show()
```

Jupyter/Colab에서는 `%matplotlib inline`을 쓰면 셀 아래에 바로 그림이 나옵니다. 이미지 처리 라이브러리는 torchvision `transforms`와 `skimage`를 상황에 맞게 섞어 쓰면 됩니다.

## 5. Knowledge Distillation (Teacher–Student)

### 아이디어

이미 잘 학습된 **크고 무거운 Teacher 모델**의 지식을 **작고 가벼운 Student 모델**에 옮깁니다(Hinton et al., 2014). Student가 정답 label만 보는 것이 아니라, Teacher가 내놓은 **확률 분포(soft label)**도 따라 하게 합니다.

- **Hard label**: 정답은 "고양이" → [0, 1, 0]
- **Soft label**: Teacher의 예측 → [0.05, 0.80, 0.15] ("호랑이와도 좀 비슷하다"는 정보가 담김)

### Temperature

Softmax 출력은 보통 한 클래스에 거의 1이 몰려 있어 soft label의 정보가 잘 드러나지 않습니다. 그래서 logit을 **temperature $$T$$로 나눠** 분포를 부드럽게 만듭니다.

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

### Loss

$$
\mathcal{L} = \alpha\cdot\text{CE}(y, \sigma(z_s)) + (1 - \alpha)\cdot T^2\cdot\text{KL}\big(\sigma(z_t / T)\ \|\ \sigma(z_s / T)\big)
$$

- **Student loss**: Student 예측과 정답 label의 Cross Entropy
- **Distillation loss**: Teacher와 Student의 soft label 사이 **KL divergence**
- $$T^2$$을 곱하는 이유: $$T$$로 나누면 기울기 크기가 $$1/T^2$$로 줄어들어서, 두 loss의 비중을 맞추기 위함
- $$\alpha$$로 두 loss의 비율 조절

```python
import torch.nn as nn
import torch.nn.functional as F

def knowledge_distillation_loss(logits, labels, teacher_logits, alpha=0.1, T=10):
    student_loss = F.cross_entropy(logits, labels)
    distillation_loss = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(logits / T, dim=1),          # KLDivLoss는 입력으로 log 확률을 받음
        F.softmax(teacher_logits / T, dim=1),
    ) * (T * T)
    return alpha * student_loss + (1 - alpha) * distillation_loss
```

### MNIST 예제

```python
class Teacher(nn.Module):              # 무거운 모델
    def __init__(self):
        super().__init__()
        self.fc1, self.bn1 = nn.Linear(784, 1200), nn.BatchNorm1d(1200)
        self.fc2, self.bn2 = nn.Linear(1200, 1200), nn.BatchNorm1d(1200)
        self.fc3 = nn.Linear(1200, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.dropout(F.relu(self.bn1(self.fc1(x))), p=0.8, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))), p=0.8, training=self.training)
        return self.fc3(x)

class Student(nn.Module):              # 가벼운 모델
    def __init__(self):
        super().__init__()
        self.fc1, self.bn1 = nn.Linear(784, 800), nn.BatchNorm1d(800)
        self.fc2, self.bn2 = nn.Linear(800, 800), nn.BatchNorm1d(800)
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

# 1) Teacher 학습 (Adam + ReduceLROnPlateau)
teacher = Teacher().to(device)
opt = torch.optim.Adam(teacher.parameters())
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=10)
# ... 30 epoch 학습 후 저장
torch.save(teacher.state_dict(), "teacher_weights.pt")

# 2) Student를 KD로 학습
teacher.load_state_dict(torch.load("teacher_weights.pt")); teacher.eval()
student = Student().to(device)
opt = torch.optim.Adam(student.parameters())

for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        t_logits = teacher(x)                     # Teacher는 업데이트하지 않음
    loss = knowledge_distillation_loss(student(x), y, t_logits)
    opt.zero_grad(); loss.backward(); opt.step()
```

`F.dropout`에 `training=self.training`을 넘겨야 `eval()`일 때 dropout이 꺼집니다.

KD로 학습한 Student는 같은 구조를 그냥 학습했을 때보다 Teacher에 가까운 정확도를 냅니다. 이 아이디어는 [BNN]({% post_url 2026-06-25-binary-neural-networks %})의 Real-to-Binary, ReActNet처럼 **실수 모델을 Teacher로, 이진 모델을 Student로** 두는 방식으로 경량화 연구에서 널리 쓰입니다.

## 6. 전부 합친 학습 스크립트 뼈대

```python
def main():
    args = parser.parse_args()
    writer = SummaryWriter(f"runs/{args.teacher}_lr{args.learning_rate}")
    model = build_model().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best = 0, 0.0
    if args.resume:
        start_epoch, best = load_checkpoint(model, optimizer, scheduler, args.save)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = train_one_epoch(model, optimizer)
        model.eval()
        val_acc = evaluate(model)
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        is_best = val_acc > best
        best = max(best, val_acc)
        save_checkpoint({"epoch": epoch, "state_dict": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(),
                         "best_top1_acc": best}, args.save)
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.save, "best.pt"))
    writer.close()
```

## 7. 정리

| 도구 | 용도 | 핵심 API |
|---|---|---|
| argparse | 실행 옵션 | `add_argument`, `parse_args`, `vars` |
| state_dict | 파라미터 저장 | `state_dict()`, `load_state_dict(strict=)` |
| Checkpoint | 학습 재개 | epoch·optimizer·scheduler를 dict로 함께 저장 |
| eval / no_grad | 추론 | Dropout·BN 동작 전환, 기울기 계산 끔 |
| TensorBoard | 학습 곡선 | `SummaryWriter`, `add_scalar`, `add_histogram` |
| Custom Transform | Augmentation | `__init__` + `__call__` |
| Knowledge Distillation | 모델 압축 | soft label, temperature, KLDivLoss |

여기까지가 AI 기초 시리즈입니다. 전체 목차는 [1편]({% post_url 2026-05-07-ml-overview %})에 있습니다.

## References

- [Python docs — argparse](https://docs.python.org/3/library/argparse.html)
- [PyTorch — Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- PyTorch Tutorials — Saving and Loading Models, TensorBoard, Writing Custom Datasets, DataLoaders and Transforms
- Hinton et al., "Distilling the Knowledge in a Neural Network," NeurIPS Workshop 2014.
- <https://github.com/liuzechun/ReActNet>
- <https://github.com/Seonghoon-Yu/AI_Paper_Review>
