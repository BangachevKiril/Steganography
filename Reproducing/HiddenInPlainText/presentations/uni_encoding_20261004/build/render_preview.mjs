import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import {PDFDocument} from 'pdf-lib';
const W='/Users/kirilbangachev/Documents/ChatGPT/Steganography/presentations/uni_encoding_20261004';
const file=W+'/output/University_Encoding_Slides.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(file));
const pdf=await PDFDocument.create();pdf.setTitle('University encoding: training and observer experiments');pdf.setSubject('Five-slide preview');pdf.setCreator('Codex');
await fs.mkdir(W+'/build/final-v4',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1.5});const data=new Uint8Array(await blob.arrayBuffer());
 await fs.writeFile(W+`/build/final-v4/slide-${i+1}.png`,data);
 const image=await pdf.embedPng(data);const page=pdf.addPage([960,540]);page.drawImage(image,{x:0,y:0,width:960,height:540});
}
await fs.writeFile(W+'/output/University_Encoding_Preview.pdf',await pdf.save());
console.log(JSON.stringify({slides:p.slides.items.length,pdf:W+'/output/University_Encoding_Preview.pdf'}));
