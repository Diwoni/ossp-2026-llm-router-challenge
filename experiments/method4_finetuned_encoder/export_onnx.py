# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""파인튜닝 인코더 라우터를 ONNX로 내보내고 동적 양자화합니다."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.common.data import write_json_atomic
from experiments.method4_finetuned_encoder.training import load_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model, _tokenizer, metadata = load_artifact(args.artifact_dir, "cpu")

    class ExportWrapper(torch.nn.Module):
        def __init__(self, router):
            super().__init__()
            self.router = router

        def forward(self, input_ids, attention_mask, dense):
            output = self.router(input_ids, attention_mask, dense)
            return output["scores"], output["costs"], output["router_embedding"]

    wrapper = ExportWrapper(model).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = args.output_dir / "router-fp32.onnx"
    int8_path = args.output_dir / "router-int8.onnx"
    batch = 2
    length = int(metadata["max_length"])
    dense_dimension = int(metadata["dense_dimension"])
    dummy = (
        torch.ones((batch, length), dtype=torch.long),
        torch.ones((batch, length), dtype=torch.long),
        torch.zeros((batch, dense_dimension), dtype=torch.float32),
    )
    with torch.no_grad():
        reference = wrapper(*dummy)
    torch.onnx.export(
        wrapper,
        dummy,
        fp32_path,
        input_names=["input_ids", "attention_mask", "dense"],
        output_names=["scores", "normalized_log_costs", "router_embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "dense": {0: "batch"},
            "scores": {0: "batch"},
            "normalized_log_costs": {0: "batch"},
            "router_embeddings": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(fp32_path))
    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"],
    )
    onnx.checker.check_model(onnx.load(int8_path))
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(int8_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    inputs = {
        "input_ids": dummy[0].numpy(),
        "attention_mask": dummy[1].numpy(),
        "dense": dummy[2].numpy(),
    }
    quantized = session.run(None, inputs)
    max_score_error = float(
        np.max(np.abs(reference[0].numpy() - quantized[0]))
    )
    max_cost_error = float(
        np.max(np.abs(reference[1].numpy() - quantized[1]))
    )
    max_router_embedding_error = float(
        np.max(np.abs(reference[2].numpy() - quantized[2]))
    )
    encoder_dir = args.artifact_dir / "encoder"
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
    ):
        source = encoder_dir / name
        if source.is_file():
            shutil.copy2(source, args.output_dir / name)
    shutil.copy2(args.artifact_dir / "metadata.json", args.output_dir / "metadata.json")
    report = {
        "artifact_type": "os-skt-method4-onnx-int8-v1",
        "fp32_bytes": fp32_path.stat().st_size,
        "int8_bytes": int8_path.stat().st_size,
        "max_dummy_score_absolute_error": max_score_error,
        "max_dummy_normalized_log_cost_absolute_error": max_cost_error,
        "max_dummy_router_embedding_absolute_error": max_router_embedding_error,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset": args.opset,
        "quantized_operations": ["MatMul", "Gemm", "Gather"],
        "source_metadata": json.loads(
            (args.artifact_dir / "metadata.json").read_text(encoding="utf-8")
        ),
    }
    write_json_atomic(args.output_dir / "export-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
