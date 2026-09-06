function initializeDocumentationControls(){
 const api=document.querySelector('#api-filter'),count=document.querySelector('#api-count');
 if(api){
  const entries=[...document.querySelectorAll('.api-entry')];
  const filter=()=>{let visible=0;for(const e of entries){e.hidden=!e.textContent.toLowerCase().includes(api.value.toLowerCase());if(!e.hidden)visible++;}count.textContent=visible+' of '+entries.length+' entries';};
  api.addEventListener('input',filter);filter();
 }
 const reveal=()=>{const el=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(el?.matches('details')){el.hidden=false;el.open=true;el.scrollIntoView();}};
 window.addEventListener('hashchange',reveal);reveal();
 const input=document.querySelector('#doc-search'),results=document.querySelector('#search-results');
 if(input){
  const render=()=>{
   results.replaceChildren();const query=input.value.toLowerCase().trim();if(query.length<2)return;
   const words=query.split(/\s+/);const found=window.DOC_SEARCH.filter(r=>words.every(w=>(r.title+' '+r.text).toLowerCase().includes(w))).slice(0,40);
   const info=document.createElement('p');info.textContent=found.length?'Showing '+found.length+' matching pages or API entries.':'No matching entries.';results.append(info);
   for(const r of found){const article=document.createElement('article'),a=document.createElement('a'),p=document.createElement('p');a.href=r.url;a.textContent=r.title;p.textContent=r.text.slice(0,200)+'…';article.append(a,p);results.append(article);}
  };
  input.value=new URLSearchParams(location.search).get('q')||'';input.addEventListener('input',render);render();
 }
 const copyIcon='<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"></rect><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"></path></svg>';
 const checkIcon='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6"></path></svg>';
 for(const pre of document.querySelectorAll('pre')){
  if(pre.closest('.code-block'))continue;
  const wrapper=document.createElement('div');wrapper.className='code-block';
  pre.before(wrapper);wrapper.append(pre);
  const button=document.createElement('button');button.type='button';button.className='copy-code';button.innerHTML=copyIcon;button.title='Copy code';button.setAttribute('aria-label','Copy code');
  button.addEventListener('click',async()=>{let label='Code copied';try{await navigator.clipboard.writeText(pre.textContent);}catch{const selection=getSelection(),range=document.createRange();range.selectNodeContents(pre);selection.removeAllRanges();selection.addRange(range);label='Code selected; press Ctrl+C';}button.innerHTML=checkIcon;button.classList.add('is-copied');button.title=label;button.setAttribute('aria-label',label);setTimeout(()=>{button.innerHTML=copyIcon;button.classList.remove('is-copied');button.title='Copy code';button.setAttribute('aria-label','Copy code');},1600);});
  wrapper.append(button);
 }
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',initializeDocumentationControls);
else initializeDocumentationControls();
