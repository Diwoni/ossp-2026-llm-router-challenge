<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Third-party data notices

## 참가팀 Method 4 모델과 런타임

선택적 Method 4 라우터는 MIT 라이선스의
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)을
리비전 `d1d99a1efae6779390caba937d92c54b5bc70e51`에 고정해 사용합니다. 공개
Train으로 파인튜닝한 router를 INT8 ONNX로 변환하고, 같은 원본의 동결 FP32
ONNX를 의미 유사도 계산에 사용합니다. 원본 가중치 SHA-256은
`1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477`입니다.

제출 런타임은 MIT의 ONNX Runtime·BSD-3-Clause의 NumPy와 protobuf·
Apache-2.0의 Hugging Face tokenizers와 FlatBuffers를 사용합니다. 정확한 제출
버전과 최종 파생 자산 SHA-256은 Docker 제출 PR에서 다시 고정합니다. 프로젝트의
Apache-2.0 라이선스는 이 제3자 모델과 런타임을 재라이선스하지 않습니다.

## 오프라인 Qwen 교사 모델

최종 Balanced 라우터의 작은 학생 head를 만들 때 Apache-2.0 라이선스의
[`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)를
오프라인 교사로만 사용했습니다. 리비전은
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 원본 가중치 SHA-256은
`0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`로
고정했습니다. 외부 학습 자료는 사용하지 않았으며 공식 공개 Train·Dev 프롬프트의
표현을 만드는 데만 사용했습니다.

Qwen 가중치와 Qwen 실행 코드는 제출 이미지에 포함되지 않습니다. 제출 런타임에는
기존 E5 임베딩과 내용 기반 피처로 Qwen 교사의 라우팅 신호를 모방하는 약 22KB의
Ridge 계수만 포함됩니다. 따라서 평가 중 네트워크 호출이나 Qwen 추론은 없습니다.

This notice applies to the adapted public prompts in
`data/train/inputs-base.json` and `data/dev/inputs-base.json`. Those files are
collections. Each source-derived part retains the license below; the project
Apache-2.0 license does not relicense third-party material. Exact revisions,
artifact hashes, and license-evidence hashes are in
`data/sources/source-pins.v1.json`.

## Belebele Korean

Source: Meta's Belebele `kor_Hang` configuration. Licensed under
[CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt). The released adaptation selects a
subset, formats passage, question, and choices as a prompt, omits answer
labels, and assigns opaque episode IDs. No endorsement is implied.

Attribution: Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe,
Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke
Zettlemoyer, and Madian Khabsa, *The Belebele Benchmark: a Parallel Reading
Comprehension Dataset in 122 Language Variants*, ACL 2024.

## CRUXEval

Copyright (c) 2023 Meta. Licensed under the [MIT License](LICENSES/MIT.txt).
The adaptation selects public examples, applies the direct input-prediction or
output-prediction prompt, omits reference inputs or outputs, and assigns opaque
episode IDs.

## GSM8K

Copyright (c) 2021 OpenAI. Licensed under the
[MIT License](LICENSES/MIT.txt). The adaptation selects public test questions,
omits solutions and answers, and assigns opaque episode IDs.

## BABILong 4K/16K components

The bAbI tasks component is copyright (c) 2015-present Facebook, Inc. and is
licensed under [BSD-3-Clause](LICENSES/BSD-3-Clause.txt). BABILong code and the
PG-19 component are licensed under [Apache-2.0](LICENSES/Apache-2.0.txt).
The adaptation uses only approved 4K and 16K configurations, adds a zero-shot task
instruction, omits targets, and assigns opaque episode IDs. Neither Facebook
nor any contributor endorses this project.

## Apache-2.0 sources

The following adapted public prompts are licensed under
[Apache-2.0](LICENSES/Apache-2.0.txt): DeepMind Mathematics, HRMCR, RuleTaker,
and TruthfulQA. Each adaptation selects an approved subset, formats only the
question-side prompt, omits gold answers and solutions, and assigns opaque
episode IDs. DeepMind Mathematics prompts are independently reproduced from
the pinned upstream generator and verified against the reference hashes in the
source record.

## Source-fetch-only material

AIME problem text is not included in this repository or release archive.
`data/train/aime-selection.json` and `data/dev/aime-selection.json` contain
only public source keys and expected prompt hashes. Users fetch the pinned
public sources and materialize those prompts locally.
