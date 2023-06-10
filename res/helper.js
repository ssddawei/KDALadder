window.$sel = document.querySelector.bind(document);
window.$sels = document.querySelectorAll.bind(document);
window.$timeString = (d) => {
  return `${d.getMonth()+1}-${d.getDate()} ${d.getHours()}:${d.getMinutes()}`;
}
window.$dateString = (d) => {
  return `${d.getFullYear()}-${(d.getMonth()+1).toString().padStart(2, '0')}-${d.getDate().toString().padStart(2, '0')}`;
}
window.$seasonString = (d) => {
  switch(d.getMonth()) {
    case 0:case 1: case 2: return `${d.getFullYear()}-season1`;
    case 3:case 4: case 5: return `${d.getFullYear()}-season2`;
    case 6:case 7: case 8: return `${d.getFullYear()}-season3`;
    case 9:case 10: case 11: return `${d.getFullYear()}-season4`;
  }
}
window.$kdaString = (kda, long) => {
  if(long)
    return `${kda.score.toFixed(1)} ${kda.kill}/${kda.death}/${kda.assist} ${(kda.win/(kda.win+kda.loss)*100).toFixed(1)}%(${kda.win}/${kda.loss})`
  else
    return `${kda.kill} / ${kda.death} / ${kda.assist}`
}
window.$f32encode = (f32) => {
  return window.btoa(String.fromCharCode.apply(null, new Uint8Array(f32.buffer.slice(0,f32.byteLength))))
}
window.$f32decode = (f32en) => {
  let blob = window.atob( f32en );
  let ary_buf = new ArrayBuffer( blob.length );
  let dv = new DataView( ary_buf );
  for( let i=0; i < blob.length; i++ ) dv.setUint8( i, blob.charCodeAt(i) );
  return new Float32Array( ary_buf );
}
window.$fetch = async function(){
  let response = await fetch.apply(window, arguments)
  if(response.status == 404) {
    return null;
  } else if(response.status != 200) {
    let message = response.statusText;
    try {
      message = (await response.json()).message;
    } catch(_) {
    }
    throw message
  }
  let toArray = arguments[1] && arguments[1].toArray;
  let toString = arguments[1] && arguments[1].toString === true;
  let text = await response.text();
  if(toString)
    return text;
  let json = toArray? `[${text}]`: text;
  if(!json) {
    return 
  } else {
    return JSON.parse(json);
  }
}
window.$queryValue = function(key) {
  let querystr = location.search.slice(1);
  let obj = {};
  querystr.split("&").forEach(i => {
    i = i.split("=");
    obj[i[0]] = i[1] || "";
  })
  return obj[key];
}
window.$throttle = function(fun, timeout, overrun) {
  let tmr;
  let reset = ()=>{ clearTimeout(tmr); tmr = null };
  {// for reset
    $throttle.__all = $throttle.__all||[];
    $throttle.__all.push(reset)
  }
  return () => {
    if(tmr) {
      overrun && overrun();
      return;
    }
    tmr = setTimeout(reset, timeout);
    return fun();
  }
}
window.$throttleResetAll = function() {
  $throttle.__all && $throttle.__all.forEach(i=>i());
}
window.$activeAnimate = function(target) {
  target.style.animation = 'none';
  target.offsetHeight; /* trigger reflow */
  target.style.animation = ''; 
}

window.$pushHistoryBack = (callback) => {
  if(!window.__pushHistoryBack) {
    window.__pushHistoryBack = []
    window.addEventListener("popstate", (e) => {
      let backCb = window.__pushHistoryBack.pop()
      backCb && backCb();
    })
  }
  window.__pushHistoryBack.push(callback);
  history.pushState('', '');
}

window.$popHistoryBack = () => {
  window.__pushHistoryBack.pop()
}

let SysDate = Date;
Date = function() {
  let args = [...arguments]
  if( typeof(args[0]) == "string" ) {
    args[0] = args[0].replace(/-/g, "/")
  }
  return new SysDate(...args)
}
Date.now = SysDate.now;

window.$prompt = async function(title){
  let tpl = `\
    <div class="dialog">\
      <div class="title">${title}</div>\
      <input type="text"></input>\
      <button class="okBtn">确认</button>\
      <button class="cancelBtn">取消</button>\
    </div>\
  `;
  let elm = document.createElement("div");
  elm.classList.add("prompt");
  elm.innerHTML = tpl;

  document.body.appendChild(elm);

  let ok, cancel;
  elm.querySelector(".okBtn").addEventListener("click", ()=>{
    ok && ok(elm.querySelector("input").value)
    elm.remove();
  })
  elm.querySelector(".cancelBtn").addEventListener("click", ()=>{
    cancel && cancel();
    elm.remove();
  })
  return await new Promise((o,x)=>{
    ok = o;
    cancel = x;
  })
}

window.$confirm = async function(title, timeout, timeoutForAccept){
  
  if(timeoutForAccept == undefined)
    timeoutForAccept = true;

  let tpl = `\
    <div class="dialog">\
      <div class="title">${title}</div>\
      <button class="okBtn">确认</button>\
      <button class="cancelBtn">取消</button>\
    </div>\
  `;
  let elm = document.createElement("div");
  elm.classList.add("prompt");
  elm.classList.add("confirm");
  elm.innerHTML = tpl;

  document.body.appendChild(elm);

  let timeoutTmr;
  let ok, cancel;
  elm.querySelector(".okBtn").addEventListener("click", ()=>{
    ok && ok(true)
    elm.remove();

    if(timeoutTmr) clearTimeout(timeoutTmr);
    timeoutTmr = null;
  })
  elm.querySelector(".cancelBtn").addEventListener("click", ()=>{
    cancel && cancel(false);
    elm.remove();

    if(timeoutTmr) clearTimeout(timeoutTmr);
    timeoutTmr = null;
  })
  if(timeout) {
    elm.querySelector(timeoutForAccept? ".okBtn": ".cancelBtn").classList.add("cooldown", "cooldown5000");
    timeoutTmr = setTimeout(()=>{
      elm.querySelector(timeoutForAccept? ".okBtn": ".cancelBtn").click()
    }, 5000)
  }
  return await new Promise((o,x)=>{
    ok = o;
    cancel = x;
  })
}

window.$alert = async function(title){
  
  let tpl = `\
    <div class="dialog">\
      <div class="title">${title}</div>\
      <button class="okBtn">确认</button>\
    </div>\
  `;
  let elm = document.createElement("div");
  elm.classList.add("prompt");
  elm.classList.add("alert");
  elm.innerHTML = tpl;

  document.body.appendChild(elm);

  let ok, cancel;
  elm.querySelector(".okBtn").addEventListener("click", ()=>{
    ok && ok()
    elm.remove();
  })
  return await new Promise((o,x)=>{
    ok = o;
    cancel = x;
  })
}