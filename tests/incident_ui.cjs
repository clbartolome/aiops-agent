// Exercise the actual central context switching and submit handlers.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
class Element {
  constructor() { this.children=[]; this.dataset={}; this.textContent=''; this.value=''; this.attrs={}; this.classList={add(){},remove(){}}; }
  append(...nodes) { for (const node of nodes) {node.parent=this;this.children.push(node);} }
  replaceChildren() { this.children=[]; }
  addEventListener(name, fn) { this[name]=fn; }
  setAttribute(k,v) { this.attrs[k]=v; }
  get lastElementChild() { return this.children.at(-1); }
  replaceWith(node) { const i=this.parent.children.indexOf(this); node.parent=this.parent;this.parent.children[i]=node; }
  querySelector(s) { if (s === 'button') return this.button ||= new Element(); return this.children.find(x=>s.includes(x.dataset.incidentId || '__none__')); }
  querySelectorAll(s) { return this.children.flatMap(x=>[...(s === '.context-select' && x.className?.split(' ').includes('context-select') ? [x] : []),...x.querySelectorAll(s)]); }
  scrollIntoView() {} focus() {} remove() { this.removed=true; }
}
const elements = Object.fromEntries(['procedure-confirmation','chat','message','history','status','procedure-chip','sessions','new-session','incidents','incident-status'].map(k=>['#'+k,new Element()]));
const incident = {id:'x', session_id:'sx', title:'High CPU <script>',message:'CPU above threshold',severity:'critical',created_at:'2026-01-01T00:00:00Z'};
const sessions = {a:{session_id:'a',title:'A',messages:[{role:'user',content:'Normal A'}]}, b:{session_id:'b',title:'B',messages:[{role:'user',content:'Normal B'}]}, sx:{session_id:'sx',messages:[{type:'incident',incident}]}};
const calls=[];
let confirmDelete=true;
let deferNextHistory=false, historyGate;
for (const session of Object.values(sessions)) Object.assign(session,{revision:0, processing:false, unread:false});
async function fetch(path, options) {
  calls.push(path);
  let data;
  if (path === '/api/sessions') data=[sessions.a,sessions.b].filter(Boolean);
  else if (path === '/api/incidents') data=sessions.sx ? [incident] : [];
  else if (path === '/api/incidents/x') { if (options?.method === 'DELETE') delete sessions.sx; else data=incident; }
  else {
    const [, , , id, action] = path.split('/');
    if (options?.method === 'DELETE') delete sessions[id];
    else if (action === 'read') { sessions[id].unread=false; data=sessions[id]; }
    else if (action === 'state') data=sessions[id];
    else if (action === 'procedure-confirmation') {sessions[id].confirmation=null; sessions[id].processing=false;sessions[id].revision++;data={type:'accepted',revision:sessions[id].revision};}
    else if (action === 'messages') {sessions[id].messages.push({role:'user',content:JSON.parse(options.body).message}); sessions[id].revision++; sessions[id].processing=id !== 'sx'; data={type:'accepted',revision:sessions[id].revision};}
    else data=sessions[id];
    if (!options && path === '/api/sessions/a' && deferNextHistory) {
      deferNextHistory=false;
      data=JSON.parse(JSON.stringify(data));
      await historyGate.promise;
    }
  }
  return {ok:true,json:async()=>data,headers:{get:()=> 'application/json'}};
}
function boot() {
  const context=vm.createContext({document:{querySelector:s=>elements[s],createElement:()=>new Element()},fetch,window:{confirm:()=>confirmDelete,alert:message=>{throw new Error(message);}},marked:{parse:text=>text},DOMPurify:{sanitize:text=>text},setTimeout(){},clearInterval(){},setInterval(){},TextDecoder});
  let script=fs.readFileSync('app/index.html','utf8').split('<script>')[1].split('</script>')[0];
  script=script.replace('    initialize();','').replace('    pollContexts();','');
  vm.runInContext(script,context);
  return code=>vm.runInContext(code,context);
}
(async()=>{
  let run=boot();
  await run('refreshSessions()'); await run('refreshIncidents()');
  assert.equal(elements['#incidents'].querySelectorAll('.context-select')[0].className,'context-select incident-item severity-critical');
  await run("selectSession('a')");
  assert.equal(elements['#history'].children[0].textContent,'> Normal A');
  await elements['#incidents'].querySelectorAll('.context-select')[0].click();
  assert.equal(elements['#history'].children.length,1);
  assert.equal(elements['#history'].children[0].className,'message incident severity-critical');
  assert.equal(elements['#history'].children[0].children[0].textContent,'INCIDENT · CRITICAL');
  assert.equal(elements['#incidents'].querySelectorAll('.context-select')[0].attrs['aria-current'],'true');
  assert(elements['#sessions'].querySelectorAll('.context-select').every(e=>e.attrs['aria-current']==='false'));
  assert.equal(elements['#chat'].hidden,false);
  elements['#message'].value='hello from incident';
  await elements['#chat'].submit({preventDefault(){}});
  assert(calls.includes('/api/sessions/sx/messages'));
  assert.equal(sessions.a.messages.length,1);
  await run("selectSession('b')");
  assert.equal(elements['#history'].children[0].textContent,'> Normal B');
  assert.equal(elements['#incidents'].querySelectorAll('.context-select')[0].attrs['aria-current'],'false');
  await run("openIncident('x')");
  assert.equal(elements['#history'].children[1].textContent,'> hello from incident');
  run=boot(); // Reload frontend state, retaining only backend histories.
  await run("openIncident('x')");
  assert.equal(elements['#history'].children.length,2);
  await run("selectSession('a')");
  assert.equal(elements['#history'].children.length,1);
  assert.equal(elements['#history'].children[0].textContent,'> Normal A');
  // Start A, navigate while processing, then independently send in B.
  elements['#message'].value='slow A';
  await elements['#chat'].submit({preventDefault(){}});
  assert.equal(elements['#chat'].button.disabled,true);
  assert(!elements['#new-session'].disabled);
  assert(elements['#sessions'].querySelectorAll('.context-select').every(e=>!e.disabled));
  await run("selectSession('b')");
  assert.equal(elements['#chat'].button.disabled,false);
  assert(elements['#sessions'].querySelectorAll('.context-select').find(e=>e.dataset.sessionId==='a').lastElementChild.className.includes('processing-loader'));
  elements['#message'].value='slow B';
  await elements['#chat'].submit({preventDefault(){}});
  assert(sessions.a.processing && sessions.b.processing);
  sessions.a.messages.push({role:'assistant',content:'Finished A'});
  Object.assign(sessions.a,{revision:3,processing:false,unread:true});
  await run('pollContexts()');
  assert(elements['#sessions'].querySelectorAll('.context-select').find(e=>e.dataset.sessionId==='a').lastElementChild.className.includes('unread-indicator'));
  assert(!elements['#history'].children.some(e=>e.innerHTML==='Finished A'));
  await run("selectSession('a')");
  assert.equal(elements['#history'].children.at(-1).innerHTML,'Finished A');
  assert.equal(sessions.a.unread,false);
  assert.equal(elements['#chat'].button.disabled,false);
  // A history response captured before an accepted submission cannot replace
  // the newer persisted snapshot while that submission is processing.
  let releaseHistory;
  historyGate={promise:new Promise(resolve=>releaseHistory=resolve)};
  deferNextHistory=true;
  const staleHistory=run('loadActive(activeContext,selectionVersion)');
  await new Promise(resolve=>setImmediate(resolve));
  elements['#message'].value='what about the deployments?';
  await elements['#chat'].submit({preventDefault(){}});
  assert.equal(elements['#history'].children.at(-1).textContent,'> what about the deployments?');
  assert.equal(elements['#chat'].button.disabled,true);
  await run('pollContexts()');
  assert.equal(elements['#history'].children.at(-1).textContent,'> what about the deployments?');
  releaseHistory();
  await staleHistory;
  assert.equal(elements['#history'].children.at(-1).textContent,'> what about the deployments?');
  assert.equal(elements['#history'].children.filter(e=>e.textContent==='> what about the deployments?').length,1);
  await run("selectSession('b')");
  await run("selectSession('a')");
  assert.equal(elements['#history'].children.at(-1).textContent,'> what about the deployments?');
  sessions.a.processing=false;
  // A processing context's current progress is restored only in its own view.
  sessions.b.progress='Running step 2';
  await run("selectSession('b')");
  assert(elements['#history'].children.some(e=>e.className==='procedure-progress'));
  await run("selectSession('a')");
  assert(!elements['#history'].children.some(e=>e.className==='procedure-progress'));
  // Confirmation controls survive navigation without duplicating the proposal.
  sessions.a.messages.push({role:'assistant',content:'Procedure found: Inspect namespace'});
  sessions.a.confirmation={run_id:'run-123'};
  sessions.a.revision++;
  await run("selectSession('a')");
  const controls=elements['#procedure-confirmation'];
  assert.equal(controls.children.length,2);
  assert.equal(controls.children[0].textContent,'Run procedure');
  assert.equal(controls.children[1].textContent,'Cancel');
  await run("selectSession('b')");
  assert.equal(controls.children.length,0);
  await run("selectSession('a')");
  assert.equal(controls.children.length,2);
  assert.equal(elements['#history'].children.filter(e=>e.innerHTML==='Procedure found: Inspect namespace').length,1);
  await controls.children[0].click();
  assert(calls.includes('/api/sessions/a/procedure-confirmation'));
  assert.equal(controls.children.length,0);
  sessions.a.confirmation={run_id:'run-456'};
  sessions.a.revision++;
  await run("selectSession('a')");
  await controls.children[1].click();
  assert.equal(controls.children.length,0);
  confirmDelete=false;
  await run("deleteContext({type:'conversation',id:'a',sessionId:'a'})");
  assert(sessions.a);
  confirmDelete=true;
  await run("deleteContext({type:'conversation',id:'a',sessionId:'a'})");
  assert(!sessions.a);
  assert.equal(elements['#history'].children[0].textContent,'> Normal B');
  await run("openIncident('x')");
  await run("deleteContext({type:'incident',id:'x',sessionId:'sx'})");
  assert(!sessions.sx);
  assert.equal(elements['#history'].children[0].textContent,'> Normal B');
  sessions.b.processing=false;
  await run("deleteContext({type:'conversation',id:'b',sessionId:'b'})");
  assert.equal(elements['#history'].children.length,0);
  assert.equal(elements['#chat'].hidden,true);
})().catch(error=>{console.error(error);process.exitCode=1;});
