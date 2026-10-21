"""Read-only frozen-sender EOS probe; all outputs go to a separate directory."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-run', type=Path, required=True)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'EMBEDDING_COMPLETE.json').exists():
        raise RuntimeError('Refusing to overwrite completed embeddings')
    import fcntl
    lock = (args.output / 'LOCK').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()

    def status(stage, **extra):
        write_json(args.output / 'STATUS.json', dict(stage=stage, updated_at=time.time(),
                   job_id=os.environ.get('SLURM_JOB_ID'), **extra))

    status('verifying_inputs')
    split = json.loads((args.split_dir / 'split_manifest.json').read_text())
    run = args.training_run
    config = json.loads((run / 'config.json').read_text())
    checkpoint = run / 'checkpoints/step_001076'
    marker = json.loads((checkpoint / 'COMPLETE.json').read_text())
    assert hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == marker['config_digest']
    assert marker['completed_update'] == split['source_checkpoint_update'] == 1076
    assert marker['config_digest'] == split['config_digest']
    assert marker['state_sha256'] == 'b6362ff7e9ab691572f8a20f9c8aa2c96675ef8f92f7c4dd67d49bbcd988a4d6'
    assert sha(checkpoint / 'COMPLETE.json') == split['source_checkpoint']['complete_sha256']
    assert sha(checkpoint / 'state.pt') == marker['state_sha256']
    source = run / 'evaluations/step_001076/UNIVERSITY_RESULTS.json'
    assert sha(source) == split['source_sha256']
    originals = {r['example_id']: r['answer'] for r in json.loads(source.read_text())['rows']}
    rows = {}
    input_hashes = {}
    for group in ['training', 'test']:
        path = args.split_dir / f'{group}_answers.jsonl'
        input_hashes[path.name] = sha(path)
        assert input_hashes[path.name] == split['files'][path.name]['sha256']
        rows[group] = read_rows(path)
        assert len(rows[group]) == 1000
        assert all(set(r) == {'example_id', 'answer'} and r['answer'] == originals[r['example_id']] for r in rows[group])
    train_ids = {r['example_id'] for r in rows['training']}
    test_ids = {r['example_id'] for r in rows['test']}
    assert len(train_ids) == len(test_ids) == 1000 and not train_ids & test_ids
    assert train_ids | test_ids == set(originals)
    del originals

    import numpy as np
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead
    assert torch.cuda.is_available() and 'L40S' in torch.cuda.get_device_name(0)
    versions = {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'trl', 'peft', 'numpy', 'tokenizers']}
    assert versions['torch'].startswith('2.4.0') and versions['transformers'] == '4.43.2' and versions['trl'] == '0.9.6'
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    asset = config['assets']['sender']['path']
    status('loading_frozen_sender')
    sender = AutoModelForCausalLMWithValueHead.from_pretrained(asset,
        peft_config=LoraConfig(r=16, lora_alpha=16, lora_dropout=.05, bias='none',
                              task_type='CAUSAL_LM', target_modules=['q_proj', 'v_proj']),
        quantization_config=None, attn_implementation='eager', torch_dtype='float16',
        local_files_only=True).to('cuda:0')
    saved = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False)
    assert saved['completed_update'] == 1076 and saved['config_digest'] == marker['config_digest']
    named = dict(sender.named_parameters())
    keys = sorted(saved['sender_trainable'])
    assert set(keys) == {n for n, parameter in named.items() if parameter.requires_grad}
    with torch.no_grad():
        for name in keys:
            value = saved['sender_trainable'][name]
            assert named[name].shape == value.shape and named[name].dtype == value.dtype, name
            named[name].copy_(value)
            assert torch.equal(named[name].detach().cpu(), value), name
    del saved
    gc.collect()
    sender.requires_grad_(False).eval()
    sender.pretrained_model.config.use_cache = False
    causal = sender.pretrained_model.get_base_model()
    backbone = causal.model
    assert causal.config.model_type == 'gemma2'
    block_count = len(backbone.layers)
    block_number = round(2 * block_count / 3)
    block_index = block_number - 1
    assert block_count == 26 and block_number == 17 and causal.config.hidden_size == 2304
    assert all(not m.training for m in sender.modules())
    assert all(not parameter.requires_grad for parameter in sender.parameters())

    def parameter_hash():
        h = hashlib.sha256()
        for name in keys:
            value = named[name].detach().contiguous().cpu()
            h.update(name.encode())
            h.update(str(value.dtype).encode())
            h.update(value.view(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    parameter_hash_before = parameter_hash()
    tokenizer = AutoTokenizer.from_pretrained(asset, local_files_only=True)
    bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    assert (bos, eos, pad) == (2, 1, 0)
    encoded = {}
    for group in rows:
        body = tokenizer([r['answer'] for r in rows[group]], add_special_tokens=False,
                         padding=False, truncation=False)['input_ids']
        assert all(not {bos, eos, pad} & set(ids) for ids in body)
        encoded[group] = [[bos] + ids + [eos] for ids in body]
        assert max(map(len, encoded[group])) <= causal.config.max_position_embeddings

    def embed(sequences, check_hidden=False):
        lengths = torch.tensor([len(s) for s in sequences], device='cuda:0')
        ids = torch.full((len(sequences), int(lengths.max())), pad, dtype=torch.long, device='cuda:0')
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor(seq, device='cuda:0')
            mask[i, :len(seq)] = 1
        positions = lengths - 1
        index = torch.arange(len(sequences), device='cuda:0')
        assert torch.all(ids[index, positions] == eos) and torch.all(mask[index, positions] == 1)
        assert torch.equal(mask.sum(-1), lengths)
        captured = []

        def capture(module, inputs, output):
            hidden = output[0]
            captured.append(hidden[index, positions].detach().float().cpu())

        handle = backbone.layers[block_index].register_forward_hook(capture)
        try:
            with torch.inference_mode():
                result = backbone(input_ids=ids, attention_mask=mask,
                    position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False,
                    output_hidden_states=check_hidden, return_dict=True)
            assert len(captured) == 1
            if check_hidden:
                expected = result.hidden_states[block_number][index, positions].float().cpu()
                assert torch.equal(captured[0], expected), 'Hook must equal residual after block17'
            result = captured[0].numpy()
            assert result.shape == (len(sequences), 2304) and np.isfinite(result).all()
            return result
        finally:
            handle.remove()

    status('checking_eos_and_batch_parity')
    probe = encoded['training'][:8]
    batched = embed(probe, check_hidden=True)
    repeated = embed(probe)
    singles = np.concatenate([embed([s]) for s in probe])
    actual_batch = embed(encoded['training'][:32])[:8]
    assert np.array_equal(batched, repeated), 'Repeated frozen forward changed'
    np.testing.assert_allclose(batched, singles, rtol=1e-2, atol=3e-2)
    np.testing.assert_allclose(batched, actual_batch, rtol=1e-2, atol=3e-2)
    validation = dict(hook_matches_post_block_hidden_state_exactly=True,
        repeated_forward_exact=True, single_vs_batch_rtol=1e-2, single_vs_batch_atol=3e-2,
        single_vs_batch_max_abs=float(np.max(np.abs(batched - singles))),
        batch8_vs_batch32_max_abs=float(np.max(np.abs(batched - actual_batch))),
        actual_batch32_parity_passed=True,
        exact_checkpoint_parameter_copy=True, all_parameters_frozen=True,
        answer_only_inputs=True, explicit_eos_positions_and_attention_masks_verified=True)
    outputs = {}
    for group in ['training', 'test']:
        chunks = []
        for start in range(0, len(rows[group]), 32):
            chunks.append(embed(encoded[group][start:start + 32]))
            status('embedding', split=group, completed=min(start + 32, 1000), total=1000)
        x = np.concatenate(chunks)
        assert x.shape == (1000, 2304) and np.isfinite(x).all()
        path = args.output / f'{group}_embeddings.npz'
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, embeddings=x,
            example_ids=np.array([r['example_id'] for r in rows[group]], dtype=np.int64),
            eos_positions=np.array([len(s)-1 for s in encoded[group]], dtype=np.int64))
        temporary.replace(path)
        outputs[path.name] = sha(path)
    assert parameter_hash() == parameter_hash_before
    assert sha(checkpoint / 'COMPLETE.json') == split['source_checkpoint']['complete_sha256']
    validation['trained_parameter_hash_unchanged_after_inference'] = True
    metadata = dict(completed_update=1076, checkpoint=str(checkpoint), checkpoint_state_sha256=marker['state_sha256'],
        checkpoint_complete_sha256=sha(checkpoint / 'COMPLETE.json'), sender_asset=config['assets']['sender'],
        trained_parameter_hash=parameter_hash_before, trained_parameter_tensors=len(keys),
        gpu=torch.cuda.get_device_name(0), model_dtype='original FP16 base with original adapter parameter dtypes',
        packages=versions, total_blocks=26, block_number_one_based=17, block_index_zero_based=16,
        hook='Gemma2DecoderLayer output[0], after MLP, post_feedforward_layernorm and residual addition; before next block normalization',
        features=2304, tokenization='explicit BOS + exact answer tokenized with add_special_tokens=False + explicit EOS; no chat template or original prompt; right padding',
        bos_token_id=bos, eos_token_id=eos, eos_token=tokenizer.eos_token, pad_token_id=pad,
        end_of_turn_token_id=tokenizer.convert_tokens_to_ids('<end_of_turn>'),
        eos_note='Appended EOS probe position, including answers originally stopped at generation cap; not the end-of-turn token.',
        batch_size=32, embedding_storage_dtype='float32 converted from model hidden-state dtype',
        split_manifest_sha256=sha(args.split_dir / 'split_manifest.json'), answer_file_sha256=input_hashes,
        extraction_script_sha256=sha(Path(__file__)), validation=validation,
        duration_seconds=time.time()-started, job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(args.output / 'EMBEDDING_COMPLETE.json', dict(status='complete', files=outputs, metadata=metadata))
    status('complete', train_examples=1000, test_examples=1000)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        print(traceback.format_exc(), flush=True)
        raise
