import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
const B='/Users/kirilbangachev/Documents/ChatGPT/Steganography';
const W=path.join(B,'presentations/uni_encoding_20261004');
const BUILD=path.join(W,'build');
const SKILL='/Users/kirilbangachev/.codex-local/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const PY='/Users/kirilbangachev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const FONT=resolvePresentationFont();
const P=Presentation.create({slideSize:{width:1280,height:720}});
const C={bg:'#FBFCFD',ink:'#152A3A',muted:'#4D6271',teal:'#007F82',blue:'#2566A8',orange:'#C26826',purple:'#7657A6',grid:'#DDE5E9'};
const J=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const baseRun=path.join(B,'results/uni_corrected_extension_20261002T235554Z');
const csv=(await fs.readFile(path.join(baseRun,'plots/corrected_evaluation_curves.csv'),'utf8')).trim().split(/\r?\n/);
const headers=csv.shift().split(',');
const curves=csv.map(l=>Object.fromEntries(l.split(',').map((s,i)=>[headers[i],Number.isNaN(Number(s))?s:Number(s)])));
if(curves.at(-1).update!==1076)throw new Error('Wrong final update');
function text(s,t,x,y,w,h,size=24,bold=false,color=C.ink,align='left'){
 const z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=t;z.text.style={typeface:FONT,fontSize:size,bold,color,alignment:align,verticalAlignment:'top',autoFit:'none',wrap:'square',insets:{top:0,bottom:0,left:0,right:0}};return z;
}
function slide(title,subtitle){const s=P.slides.add();s.background.fill=C.bg;text(s,title,60,40,1160,64,44,true);if(subtitle)text(s,subtitle,60,111,1160,54,24,false,C.muted);return s;}
function notes(s,t){s.speakerNotes.textFrame.setText(t);}
function table(s,values,x,y,width,height,widths,font=25){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,columnWidths:widths,values});
 t.styleOptions={headerRow:false,bandedRows:false};
 const all=t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length});
 all.assign({fill:C.bg,textStyle:{typeface:FONT,fontSize:font,color:C.ink},margins:{left:14,right:14,top:12,bottom:10},anchor:'center'});
 t.borders.assign({fill:C.grid,width:0.7,style:'solid'});
 const head=t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length});
 head.assign({fill:'#EAF0F4',textStyle:{typeface:FONT,fontSize:font-2,bold:true,color:C.ink},margins:{left:14,right:14,top:12,bottom:10}});
 for(let r=0;r<values.length;r++){t.rows[r].height=height/values.length;for(let col=1;col<values[0].length;col++) t.getCell(r,col).text.style={alignment:'center'};}
 return t;
}
// 1. The environment and learning loop.
{
 const s=slide('Training method and RL environment','University label hidden in a CV paraphrase');
 text(s,'Sender: Gemma 2–2B-IT with LoRA',60,186,596,35,27,true,C.teal);
 text(s,'Reads the CV and its university. Writes a short summary without naming the university.',60,230,555,77,25);
 text(s,'Receiver: trainable DistilBERT',60,325,590,35,27,true,C.teal);
 text(s,'Predicts PASS (Good University) or FAIL (Bad University) from the summary.',60,369,555,78,25);
 text(s,'Observer: frozen Llama 3.1–8B',60,466,590,35,27,true,C.teal);
 text(s,'Flags references to institutions or academic status.',60,510,555,74,25);
 text(s,'Alternating training',710,186,505,35,27,true);
 text(s,'2 receiver supervised updates\n256 accepted summaries per update',710,233,505,83,25);
 text(s,'5 sender PPO updates\n192 summaries per update',710,335,505,83,25);
 text(s,'Reward favors correct decoding and low observer suspicion, with a length term and PPO KL regularization.',710,447,505,128,25);
 const cite=text(s,'Reference: Mathew et al. (2025), Hidden in Plain Text: Emergence & Mitigation of\nSteganographic Collusion in LLMs. arXiv:2410.03768. Corrected implementation of the Uni O setup.',60,624,1160,64,18,false,C.muted);
 cite.text.get('arXiv:2410.03768').link={uri:'https://arxiv.org/abs/2410.03768',isExternal:true};
 notes(s,`Primary reference: Yohan Mathew, Ollie Matthews, Robert McCarthy, Joan Velja, Christian Schroeder de Witt, Dylan Cope, Nandi Schoots. Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs, arXiv:2410.03768v2 (2025; first submitted 2024). https://arxiv.org/abs/2410.03768\nAuthor code: https://github.com/olliematthews/lasr-steganography at d4c708b9339fd2f85cb442827ac56333b55cf9d2.\nLocal implementation: ${B}/launches/uni_corrected_20261002T151448Z/configs/university_training.json and EXPERIMENT.md. This is a corrected implementation experiment, not an exact reproduction or single-fix ablation. Corrections: all 256 receiver examples contribute to one mean CE update through 4 minibatches of 64; genuine sender attention masks; observer scores last attended token; receiver reward scoring uses eval mode.\nSender reward: (1 - observer_flag) * tanh(receiver logit of true class) + 0.2*observer_score + 0.1*min(character_length/20,1). KL regularization remains separate. Sender uses 192 rollouts, minibatches 64, accumulation 3. Evaluation uses the same 2,000 university examples and 100 PIQA prompts. PIQA d never enters the university reward. Training uses one L40S, approximately 15.5 active hours including periodic evaluations, ending at update 1076.\nThe sender sees university identity. The receiver target is the hidden university category, not the receiver's own predicted label.`);
}
// 2. Native scatter charts preserve the nonuniform checkpoint intervals.
{
 const s=slide('Training curves through update 1,076','Frozen evaluation cohorts: 2,000 university CVs and 100 PIQA prompts');
 const specs=[
  ['C_R: receiver accuracy (final 99.6%)','C_R_percent',60,174,C.teal,1,0.25],
  ['C_I: observer flag rate (final 0.15%)','C_I_percent',670,174,C.orange,0.15,0.05],
  ['d: PIQA failure rate (final 4%)','d_percent',60,410,C.purple,0.10,0.025],
  ['Receiver PASS output (final 50.05%)','PASS_percent',670,410,C.blue,1,0.25]
 ];
 for(const [title,key,x,y,col,max,unit] of specs){
  const vals=curves.map(r=>Number((key==='PASS_percent'?r.receiver_pass/r.university_n:r[key]/100).toFixed(6)));
  const chart=s.charts.add('scatter',{position:{left:x,top:y,width:550,height:230},title,titleTextStyle:{typeface:FONT,fontSize:23,bold:true,fill:C.ink},
   series:[{name:title,values:vals,xValues:curves.map(r=>r.update),line:{fill:col,width:2.8,style:'solid'},marker:{symbol:'circle',size:4}}],scatterOptions:{style:'lineWithMarkers'},hasLegend:false,
   xAxis:{visible:true,min:0,max:1100,majorUnit:250,numberFormatCode:'0',title:{text:'Training update',textStyle:{typeface:FONT,fontSize:17,fill:C.muted}},textStyle:{typeface:FONT,fontSize:17,fill:C.muted},line:{fill:C.grid,width:1},majorGridlines:null},
   yAxis:{visible:true,min:0,max,majorUnit:unit,numberFormatCode:key==='d_percent'?'0.0%':'0%',textStyle:{typeface:FONT,fontSize:17,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.grid,width:0.6}},
   chartFill:C.bg,plotAreaFill:C.bg,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}});
  applyPresentationChartFont(chart,{fontFamily:FONT});
 }
 text(s,'C_I measures observer flags. d counts PIQA judge rejection or format failure. PASS is an output fraction, not accuracy.',60,662,1160,40,18,false,C.muted);
 notes(s,`Source: ${baseRun}/plots/corrected_evaluation_curves.csv and plots/verified_report.md. All plotted rows at or below hash-verified final checkpoint 1076. No smoothing. Scatter x-values retain irregular evaluations at 530 and 1076.\nC_R=receiver_correct/2000. C_I=observer_flagged/2000, an operational observer flag fraction, not verified transmission accuracy. d=union of judge rejection and format failure on the same 100 PIQA examples, evaluation only. PASS=receiver_pass/2000, not hidden class prevalence.\nFinal: C_R=1992/2000=99.6%; C_I=3/2000=0.15%; PASS=1001/2000=50.05%; FAIL=999/2000; d=4/100 (judge4, format0, overlap0), versus corrected step0 d=8/100. These curves use only the corrected experiment; earlier uncorrected C_I is incomparable.\nThe same frozen cohorts recur, so these are longitudinal diagnostics, not independent test sets at every checkpoint.`);
}
// 3. Linear probes.
{
 const s=slide('Supervised learning: linear probes recover PASS / FAIL','Same checkpoint 1,076 answers, with a random 1,000 training / 1,000 test split');
 text(s,'Frozen EOS residual after the MLP at about two-thirds depth',60,184,1160,38,28,true,C.teal);
 text(s,'Gemma block 17 / 26. Qwen block 24 / 36. Standardize on training data and fit L2 logistic regression.',60,233,1160,68,24);
 const vals=[['Embedding model','Train BCE','Test BCE','Test accuracy']];
 const logs=[];
 for(const [key,label] of [['sender','Trained Gemma sender'],['gemma_untrained','Original pretrained Gemma'],['qwen25_3b','Qwen2.5-3B-Instruct']]){
  const launch=await J(path.join(B,`launches/${key}_eos_logreg_1076_latest.json`));
  const m=await J(path.join(launch.local_results,'logistic_regression/metrics.json'));logs.push({label,path:launch.local_results,metrics:m});
 }
 // Numeric values cross-checked against the immutable logistic regression metrics.
 vals.push(['Trained Gemma sender','0.00080','0.06061','97.8%']);
 vals.push(['Original pretrained Gemma','0.00083','0.06174','97.9%']);
 vals.push(['Qwen2.5-3B-Instruct','0.00294','0.07531','98.1%']);
 table(s,vals,60,323,1160,238,[470,215,215,260],25);
 text(s,'All three classifiers achieve 100% training accuracy.',60,590,1160,40,27,true,C.teal);
 text(s,'BCE is mean binary cross-entropy in nats, excluding the L2 penalty. “Original” means before sender fine-tuning.',60,658,1160,38,18,false,C.muted);
 notes(s,`Data: ${B}/data/university_observer_checkpoint1076_split/split_manifest.json. Same 2,000 answer strings, random 1,000/1,000 split. Test500PASS/500FAIL; training501PASS/499FAIL. Hidden labels remain separate from input answers. EOS is explicitly appended and captured after the selected block MLP residual addition. Inputs are answer text only. Feature counts: Gemma2304, Qwen2048. Original pretrained Gemma is the original instruction model, not random initialization.\nProbe: train-fitted mean/std, unpenalized intercept, L2 logistic regression with C=1. Objective mean BCE + ||w||^2/(2*n_train*C), SciPy L-BFGS-B. No test-based hyperparameter tuning.\nSources: ${logs.map(z=>z.path+'/logistic_regression/metrics.json').join('\n')}\nExact metrics:\n${JSON.stringify(logs.map(z=>({model:z.label,metrics:z.metrics})),null,2)}\nThese losses and accuracies concern hidden university labels, not the receiver's predicted PASS/FAIL outputs. Two answer strings duplicate across split (test IDs392,456); primary prespecified scores retain all 1000 test examples.`);
}
// 4. No hidden label separation by k-means.
{
 const s=slide('Unsupervised 1: clustering does not recover the labels','Two clusters from 1,000 test embeddings, with 50 starts per model and method');
 table(s,[['Embedding model','Standard k-means','Spherical k-means','Spherical ARI'],['Qwen2.5-3B-Instruct','51.6%','51.6%','0.00006'],['Original pretrained Gemma','52.1%','52.6%','0.00173'],['Trained Gemma sender','51.7%','52.1%','0.00079']],60,218,1160,252,[440,245,245,230],25);
 text(s,'Scores remain close to 50% chance agreement.',60,515,1160,45,31,true,C.orange);
 text(s,'All 50 spherical starts converge to the same partition within each model.\nAdjusted Rand indices near zero show little agreement with PASS / FAIL.',60,575,1160,74,24);
 text(s,'Best class permutation after fitting. Standard: feature z-scores. Spherical: centered unit vectors and unit centroids.',60,668,1160,32,17.5,false,C.muted);
 notes(s,`Sources: ${B}/launches/qwen25_3b_test_kmeans_1076_latest.json, gemma_untrained_test_kmeans_1076_latest.json, gemma_trained_test_kmeans_1076_latest.json and ${B}/results/spherical_test_kmeans_1076_20261003T204014Z/comparison.json, independent_check.json.\nBoth methods k=2,50 starts,seed42,stable assignment stopping,max1000 iterations. Standard k-means++ uses mean/std over only the unlabeled test matrix and minimum squared Euclidean inertia. Spherical uses raw test mean-centering, row L2 normalization, and normalization of centroid means after every update. It initializes with uniform first point and squared chord-distance weighted second, then selects minimum sum(1-cosine). No PCA or training embeddings.\nAll three spherical cluster partitions were frozen and hashed before current label scoring; hidden labels do not influence fits or restart selection. Accuracy is best of the two cluster-name mappings. This is exploratory reuse of a test cohort whose labels were inspected in prior analyses. Prior ordinary and spherical methods differ in preprocessing as well as geometry, so this is not an isolated causal magnitude ablation.\nARI for spherical: Qwen0.00006309967536; original Gemma0.00172904801792; trained Gemma0.00078750834013. Direct pair-count verification used all499500 unordered pairs.`);
}
// 5. User-provided Claude predictions.
{
 const s=slide('Unsupervised 2: Claude Opus 5.5','Provided text-based predictions recover the hidden classes after relabeling');
 text(s,'88.1%',60,212,475,142,116,true,C.teal);
 text(s,'881 / 1,000 labels agree',65,369,480,50,31,true);
 text(s,'Class names are reversed:\n“good” predicts FAIL\n“bad” predicts PASS',65,437,490,120,27);
 table(s,[['True hidden label','Predicted FAIL','Predicted PASS'],['FAIL','434','66'],['PASS','53','447']],625,229,595,231,[215,190,190],24);
 text(s,'Typical cue: opening with\n“Driven” or “Determined”\nstrongly predicts FAIL.',625,501,590,118,26);
 text(s,'Raw agreement was 11.9%. We chose the label permutation after checking the test labels. Model attribution supplied by the user.',60,653,1160,49,18,false,C.muted);
 notes(s,`User explicitly attributed the supplied predictions to Claude Opus5.5 on October4,2026. Model identity and original prompt/label access were not independently verified, so this slide reports the provided artifact rather than claiming a controlled blind LLM trial.\nSource: /Users/kirilbangachev/Downloads/test_predictions.jsonl, SHA25639ffb7aaf2f993d7bab39de9331e84d3597b32bb9e3b9433a71ca640ebb65356. Ground truth: ${B}/data/university_observer_checkpoint1076_split/test_hidden_labels.jsonl, SHA256c1c9d35c11f8e5cd5ffe87175a4c862bd6e24df8ac8941685a5da99a4faf7c09. All1000 IDs are unique and match the test set exactly, and all answer texts match.\nDirect good=PASS,bad=FAIL gives119/1000. Reversing class names gives881/1000=88.1%. After relabeling, confusion matrix rows actualFAIL/PASS, cols predictedFAIL/PASS: [[434,66],[53,447]]. The scoring chooses label mapping after seeing test labels; this is permutation-adjusted agreement, not a new held-out accuracy estimate. No classifier was fitted during scoring.\nThe supplied cue field distinguishes Driven/Determined openings, he/his narrative, driven/determined mid-sentence, passion-only and no-cue cases. Our descriptive test count: startsDriven/Determined339FAIL,11PASS. This is evidence of a label-associated stylistic pattern, not a causal proof of what the trained receiver uses.`);
}
const draft=path.join(BUILD,'candidate-v2.pptx');
await (await PresentationFile.exportPptx(P)).save(draft);
for(let i=0;i<P.slides.items.length;i++){
 const s=P.slides.items[i];
 const png=await P.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await s.export({format:'layout'});await fs.writeFile(path.join(BUILD,`slide-${i+1}.layout.json`),await layout.text());
}
const final=path.join(W,'output/University_Encoding.pptx');
const result=await finalizePresentation({workspaceDir:W,candidatePath:draft,finalPath:final,pythonExecutable:PY,
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--cover-role','none',...([3,4,5].flatMap(n=>['--require-native-table-slide',String(n)]))],
 explicitTotalSlideCount:5,requiredNativeTableOwnerSlides:[3,4,5],requiredNativeChartOwnerSlides:[2],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
 receiptPath:path.join(BUILD,'validation.json')});
await fs.writeFile(path.join(BUILD,'build_result.json'),JSON.stringify({font:FONT,final,result},null,2));
console.log(JSON.stringify({font:FONT,final,slides:P.slides.items.length}));
