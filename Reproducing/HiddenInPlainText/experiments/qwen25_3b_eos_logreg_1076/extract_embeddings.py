"""Independent pretrained Qwen2.5-3B-Instruct EOS control on the same trained-sender answers."""
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
    from transformers import AutoTokenizer, AutoModelForCausalLM
    assert torch.cuda.is_available() and 'L40S' in torch.cuda.get_device_name(0)
    versions = {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'trl', 'peft', 'numpy', 'tokenizers']}
    assert versions['torch'].startswith('2.4.0') and versions['transformers'] == '4.43.2' and versions['trl'] == '0.9.6'
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    asset_manifest = json.loads(Path(__file__).with_name('embedding_asset.json').read_text())
    assert asset_manifest['repo'] == 'Qwen/Qwen2.5-3B-Instruct'
    assert asset_manifest['revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
    asset = asset_manifest['path']
    for name, expected in asset_manifest['files_sha256'].items():
        assert Path(name).name == name
        assert sha(Path(asset) / name) == expected
    status('loading_pretrained_qwen25_3b_instruct')
    causal = AutoModelForCausalLM.from_pretrained(asset,
        quantization_config=None, attn_implementation='eager', torch_dtype='float16',
        local_files_only=True).to('cuda:0')
    assert causal.__class__.__name__ == 'Qwen2ForCausalLM'
    assert not getattr(causal, 'peft_config', None)
    assert not getattr(causal, '_hf_peft_config_loaded', False)
    named = dict(causal.named_parameters())
    keys = sorted(named)
    assert not any('lora_' in name for name in keys)
    assert not any('lora_' in name for name, _ in causal.named_modules())
    causal.requires_grad_(False).eval()
    causal.config.use_cache = False
    backbone = causal.model
    assert causal.config.model_type == 'qwen2'
    block_count = len(backbone.layers)
    block_number = round(2 * block_count / 3)
    block_index = block_number - 1
    assert block_count == 36 and block_number == 24 and causal.config.hidden_size == 2048
    assert all(not m.training for m in causal.modules())
    assert all(not parameter.requires_grad for parameter in causal.parameters())

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
    assert (bos, eos, pad) == (None, 151645, 151643)
    assert tokenizer.eos_token == '<|im_end|>' and causal.config.eos_token_id == eos
    assert causal.config.bos_token_id == 151643
    assert not tokenizer.init_kwargs.get('add_bos_token', False)
    encoded = {}
    for group in rows:
        body = tokenizer([r['answer'] for r in rows[group]], add_special_tokens=False,
                         padding=False, truncation=False)['input_ids']
        assert all(not set(tokenizer.all_special_ids) & set(ids) for ids in body)
        assert body == tokenizer([r['answer'] for r in rows[group]], add_special_tokens=True,
                                 padding=False, truncation=False)['input_ids']
        encoded[group] = [ids + [eos] for ids in body]
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
                assert torch.equal(captured[0], expected), 'Hook must equal residual after block24'
            result = captured[0].numpy()
            assert result.shape == (len(sequences), 2048) and np.isfinite(result).all()
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
        pretrained_model_without_adapters=True, all_parameters_frozen=True,
        answer_only_inputs=True, explicit_eos_positions_and_attention_masks_verified=True)
    outputs = {}
    for group in ['training', 'test']:
        chunks = []
        for start in range(0, len(rows[group]), 32):
            chunks.append(embed(encoded[group][start:start + 32]))
            status('embedding', split=group, completed=min(start + 32, 1000), total=1000)
        x = np.concatenate(chunks)
        assert x.shape == (1000, 2048) and np.isfinite(x).all()
        path = args.output / f'{group}_embeddings.npz'
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, embeddings=x,
            example_ids=np.array([r['example_id'] for r in rows[group]], dtype=np.int64),
            eos_positions=np.array([len(s)-1 for s in encoded[group]], dtype=np.int64))
        temporary.replace(path)
        outputs[path.name] = sha(path)
    assert parameter_hash() == parameter_hash_before
    assert sha(checkpoint / 'COMPLETE.json') == split['source_checkpoint']['complete_sha256']
    validation['model_parameter_hash_unchanged_after_inference'] = True
    asset_hashes = {name: sha(Path(asset) / name) for name in asset_manifest['files_sha256']}
    assert asset_hashes == asset_manifest['files_sha256']
    metadata = dict(source_checkpoint_update=1076, source_checkpoint=str(checkpoint), source_checkpoint_state_sha256=marker['state_sha256'],
        source_checkpoint_complete_sha256=sha(checkpoint / 'COMPLETE.json'), embedding_asset={k: asset_manifest[k] for k in ['repo', 'revision', 'path']},
        embedding_model_kind='pretrained_qwen25_3b_instruct',
        no_sender_adapters_loaded=True, sender_checkpoint_weights_loaded=False,
        interpretation='Separate pretrained Qwen2.5-3B-Instruct with no sender fine-tuning or adapters. Answer text still comes from sender checkpoint1076.',
        model_parameter_hash=parameter_hash_before, model_parameter_tensors=len(keys),
        pretrained_asset_file_sha256=asset_hashes,
        gpu=torch.cuda.get_device_name(0), model_dtype='torch.float16',
        packages=versions, total_blocks=36, block_number_one_based=24, block_index_zero_based=23,
        hook='Qwen2DecoderLayer output[0], after MLP and residual addition; before next block normalization',
        features=2048, tokenization='exact answer tokenized with add_special_tokens=False + explicit native EOS; no BOS, chat template or original prompt; right padding',
        bos_token_id=bos, eos_token_id=eos, eos_token=tokenizer.eos_token, pad_token_id=pad,
        model_config_bos_token_id=causal.config.bos_token_id, bos_prepended=False,
        eos_note='Appended native EOS <|im_end|> probe position, including answers originally stopped at generation cap; for Qwen Instruct EOS is also the chat end-of-turn token.',
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
