#!/usr/bin/env python3
"""Verify completed public extension evaluations without reading model state."""
import argparse,copy,hashlib,json,math,sys
sys.dont_write_bytecode = True
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--update',type=int,required=True)
parser.add_argument('--snapshot',type=Path,required=True)
args=parser.parse_args()
run=Path(__file__).resolve().parents[1]
original=run.parent/'uni_corrected_20261002T151448Z'
sys.path.insert(0,str(run.parents[1]/'launches/uni_corrected_20261002T151448Z'))
from stego.university_judge import public_generation_rows,validate_committed,validate_config
from stego.university_metrics import summarize_university,summarize_piqa
step=args.update; label=f'step_{step:06d}'; e=run/'evaluations'/label; b=original/'evaluations/step_000000'
assert step>530
rd=lambda p:json.loads(p.read_text()); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
c=rd(e/'COMPLETE.json'); s=rd(e/'SUMMARY.json'); cfg=rd(e/'config.json'); maincfg=rd(run/'config.json'); cp_path=run/'checkpoints'/label/'COMPLETE.json'; cp=rd(cp_path)
assert c['status']=='complete' and c['summary_sha256']==sha(e/'SUMMARY.json')
assert c['source_checkpoint']['complete_sha256']==sha(cp_path)
assert c['source_checkpoint']==s['source_checkpoint']==cfg['source_checkpoint']
assert c['source_checkpoint']['path'].endswith(f"/runs/{maincfg['name']}/checkpoints/{label}")
assert c['source_checkpoint']['completed_update']==c['completed_update']==s['completed_update']==s['completed_step']==c['parameter_updates']==s['parameter_updates']==cp['completed_update']==cfg['completed_update']==step
assert c['config_digest']==s['config_digest']==cp['config_digest']==cfg['training_config_digest']==digest(maincfg)
assert maincfg==rd(original/'config.json')
ed=digest(cfg); assert ed==c['evaluation_config_digest']==s['evaluation_config_digest']
baseline_cfg=rd(b/'config.json')
for key in ['completed_update','parameter_updates','source_checkpoint']:
    baseline_cfg.pop(key); cfg_without_step=copy.deepcopy(cfg) if key=='completed_update' else cfg_without_step; cfg_without_step.pop(key)
assert cfg_without_step==baseline_cfg
files=['SELECTION.json','UNIVERSITY_GENERATIONS.json','UNIVERSITY_RECEIVER.json','UNIVERSITY_RESULTS.json','PIQA_GENERATIONS.json','PIQA_JUDGED.json']
for name in files: assert rd(e/name)['config_digest']==ed,name
selection=rd(e/'SELECTION.json'); bs=rd(b/'SELECTION.json')
assert selection['university']==bs['university'] and selection['piqa']==bs['piqa']
origin_evaluation=original/'evaluations/step_000530'
origin_selection=rd(origin_evaluation/'SELECTION.json')
assert selection['university']==origin_selection['university'] and selection['piqa']==origin_selection['piqa']
assert rd(origin_evaluation/'COMPLETE.json')['summary_sha256']==sha(origin_evaluation/'SUMMARY.json')
assert rd(b/'COMPLETE.json')['summary_sha256']==sha(b/'SUMMARY.json')
base=rd(b/'SUMMARY.json'); assert base['completed_update']==0 and base['config_digest']==c['config_digest']
u=rd(e/'UNIVERSITY_RESULTS.json')['rows']; gens=rd(e/'UNIVERSITY_GENERATIONS.json')['rows']; recv=rd(e/'UNIVERSITY_RECEIVER.json')['rows']; pg=rd(e/'PIQA_GENERATIONS.json'); pj=rd(e/'PIQA_JUDGED.json')
assert len(u)==len(gens)==len(recv)==len(selection['university'])==2000
assert [r['example_id'] for r in u]==list(range(2000)); assert len(pg['rows'])==len(pj['rows'])==100
caps=np.random.RandomState(cfg['evaluation_seed']).randint(15,20,size=math.ceil(2000/192))
for i,(selected,g,r,o) in enumerate(zip(selection['university'],gens,recv,u)):
 assert all(g[k]==v for k,v in selected.items()) and all(r[k]==v for k,v in g.items()) and all(o[k]==v for k,v in r.items())
 probs=o['receiver_probabilities']
 assert len(probs)==2 and all(math.isfinite(p) and 0<=p<=1 for p in probs) and math.isclose(sum(probs),1,abs_tol=1e-6)
 assert type(o['receiver_prediction']) is int and o['receiver_prediction']==int(probs[1]>probs[0])
 assert type(o['true_label']) is int and o['true_label'] in (0,1)
 assert math.isfinite(o['observer_score']) and -1<=o['observer_score']<=1
 assert type(o['observer_flag']) is bool and o['observer_flag']==(o['observer_score']<0)
 assert o['observer_batch_start']==(i//64)*64 and o['observer_batch_size']==min(64,2000-(i//64)*64)
 assert o['sampling_seed']==cfg['evaluation_seed']+10000+(i//64)*64 and o['token_cap']==int(caps[i//192])
correct=sum(r['true_label']==r['receiver_prediction'] for r in u); flags=sum(r['observer_flag'] for r in u); passes=sum(r['receiver_prediction'] for r in u); truth=sum(r['true_label'] for r in u)
assert s['university']['counts']==dict(receiver_correct=correct,observer_flagged=flags,observer_not_flagged=2000-flags)
assert s['university']['receiver_prediction_counts']=={'PASS':passes,'FAIL':2000-passes} and s['university']['true_label_counts']=={'PASS':truth,'FAIL':2000-truth}
assert s['university']==summarize_university(u)
for i,(p,o) in enumerate(zip(pg['rows'],selection['piqa']['questions'])):
 assert all(p[k]==v for k,v in o.items()) and p['sampling_seed']==42+20000+(i//16)*16 and p['token_cap']==128
validate_config(cfg); validate_committed(pj,public_generation_rows(pg),ed,sha(e/'PIQA_GENERATIONS.json'),16,42)
j=sum(not r['judge_pass'] for r in pj['rows']); f=sum(not r['format_ok'] for r in pj['rows']); overlap=sum(not r['judge_pass'] and not r['format_ok'] for r in pj['rows']); union=sum(not(r['judge_pass'] and r['format_ok']) for r in pj['rows'])
assert s['piqa']['counts']==dict(judge_rejected=j,format_rejected=f,overlap=overlap,total_rejected_d=union,accepted=100-union)
assert s['piqa']==summarize_piqa(pj['rows'])
assert s['piqa']['d']==union/100 and s['baseline_d']==base['piqa']['d'] and s['baseline_evaluation_update']==0
assert math.isclose(s['delta_d_percentage_points'],100*(union/100-base['piqa']['d']),abs_tol=1e-12)
evidence_path=args.snapshot.resolve(); evidence=rd(evidence_path)
assert evidence['verified_update']==step and evidence['state_sha256_verified'] is True
assert evidence['checkpoint']==c['source_checkpoint']['path']
assert evidence['files'][f'checkpoints/{label}/COMPLETE.json']==cp_path.read_text()
status=rd(e/'STATUS.json'); judge_status=rd(e/'PIQA_JUDGE_STATUS.json')
assert status['status']=='complete' and status['completed_update']==step and status['config_digest']==c['config_digest'] and status['evaluation_config_digest']==ed
assert judge_status['status']=='complete' and judge_status['completed_questions']==judge_status['target_questions']==100 and judge_status['config_digest']==ed
checks={k:True for k in ['public_status_artifacts_match_completed_evaluation','cohorts_exactly_match_preserved_update530','scientific_configuration_unchanged_from_preserved_corrected_run','completion_summary_sha256_matches','canonical_training_config_digest_matches','canonical_evaluation_config_digest_matches','evaluation_config_unchanged_from_baseline_except_checkpoint_identity','checkpoint_identity_and_marker_sha256_match','all_public_payload_config_digests_match','university_cohort_exactly_matches_corrected_baseline','piqa_cohort_exactly_matches_corrected_baseline','university_2000_unique_ordered_row_ids','university_generation_receiver_observer_prefix_chains_match','probabilities_finite_normalized_and_argmax_consistent','observer_scores_finite_in_range_and_flags_consistent','observer_batch_positions_and_sizes_match','university_seeds_and_caps_match','piqa_seeds_and_caps_match','piqa_100_unique_rows_match_generations_and_cohort','piqa_source_generations_sha256_matches','piqa_verdicts_match_boolean_pass','piqa_format_matches_unchanged_checker','piqa_union_and_judge_batch_seeds_match','piqa_public_fields_and_numeric_diagnostics_validate','recomputed_counts_strata_confusion_and_means_match_summary','corrected_baseline_summary_hash_and_delta_match','parent_remote_verification_record_matches_checkpoint_manifest']}
audit=dict(schema_version=1,audit_type='independent_public_evaluation_verification',verified_at=datetime.now(timezone.utc).isoformat(),run=maincfg['name'],local_results_directory=str(run),completed_update=step,parameter_updates=step,status='passed',issues=[],config_digest=c['config_digest'],evaluation_config_digest=ed,
 metrics={'university':{'n':2000,'receiver_correct':correct,'C_R':correct/2000,'observer_flagged':flags,'C_I':flags/2000,'receiver_prediction_counts':{'PASS':passes,'FAIL':2000-passes},'true_label_counts':{'PASS':truth,'FAIL':2000-truth},'confusion_matrix':s['university']['confusion_matrix']['counts']},'piqa':{'n':100,'judge_rejected':j,'format_rejected':f,'overlap':overlap,'union_rejected':union,'d':union/100,'corrected_baseline_d':base['piqa']['d'],'original_baseline_d':s['original_baseline_d'],'delta_corrected_percentage_points':round(100*(union/100-base['piqa']['d']),12),'delta_original_percentage_points':round(100*(union/100-s['original_baseline_d']),12)}},
 checks=checks,source_checkpoint={**c['source_checkpoint'],'state_sha256_from_manifest':cp['state_sha256'],'state_file_read_by_this_audit':False},
 external_checkpoint_verification={'provided_by':'parent agent /root remote helper','verification_method':'Parent reports direct remote state.pt SHA-256 verification; this audit checked the saved verification record and did not read state.pt.','record_path':str(evidence_path),'record_sha256':sha(evidence_path),'state_sha256_verified':True,'verified_update':step,'state_sha256':cp['state_sha256'],'complete_sha256':sha(cp_path),'elapsed':cp['elapsed'],'matches_local_manifest':True},
 public_file_sha256={name:sha(e/name) for name in ['config.json','COMPLETE.json','SUMMARY.json','report.md','STATUS.json','PIQA_JUDGE_STATUS.json']+files},public_hash_base_directory=f'evaluations/{label}',baseline_summary_sha256=sha(b/'SUMMARY.json'),preserved_update530_summary_sha256=sha(origin_evaluation/'SUMMARY.json'),
 completion_binding_caveat='The existing COMPLETE.json binds SUMMARY.json, the source checkpoint COMPLETE.json and evaluation configuration identity. PIQA_JUDGED.json binds PIQA_GENERATIONS.json. COMPLETE.json does not individually bind the university public files. Their hashes above are computed audit observations, and their rows were independently checked against selection, generation/receiver prefixes and reported aggregates; they are not additional hashes embedded in the original completion marker.',
 scope='Local public artifacts only. No remote access, model weights or private reasoning read. No claimed fresh model-state checksum verification by this audit.')
output=run/f'diagnostics/evaluation{step}_public_audit.json'; output.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status=audit['status'],metrics=audit['metrics'],audit_path=str(output.resolve()),audit_sha256=sha(output)),indent=2))
