let $sel = document.querySelector.bind(document);
let $sels = document.querySelectorAll.bind(document);
let $timeString = (d) => {
  return `${d.getMonth()+1}-${d.getDate()} ${d.getHours()}:${d.getMinutes()}`;
}
let $dateString = (d) => {
  return `${d.getFullYear()}-${d.getMonth()+1}-${d.getDate()}`;
}
let $seasonString = (d) => {
  switch(d.getMonth()) {
    case 0:case 1: case 2: return `${d.getFullYear()}-season1`;
    case 3:case 4: case 5: return `${d.getFullYear()}-season2`;
    case 6:case 7: case 8: return `${d.getFullYear()}-season3`;
    case 9:case 10: case 11: return `${d.getFullYear()}-season4`;
  }
}
let $kdaString = (kda, long) => {
  if(long)
    return `${kda.score.toFixed(1)} ${kda.kill}/${kda.death}/${kda.assist} ${(kda.win/(kda.win+kda.loss)*100).toFixed(1)}%(${kda.win}/${kda.loss})`
  else
  return `${kda.kill} / ${kda.death} / ${kda.assist}`
}
let $fetch = async function(){
  let response = await fetch.apply(window, arguments)
  if(response.status == 404){
    return null;
  }
  
  let text = await response.text();
  let json = '[' + text + ']';
  return JSON.parse(json);
}

let $prompt = async function(title){
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

let $confirm = async function(title){
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

  let ok, cancel;
  elm.querySelector(".okBtn").addEventListener("click", ()=>{
    ok && ok(true)
    elm.remove();
  })
  elm.querySelector(".cancelBtn").addEventListener("click", ()=>{
    cancel && cancel(false);
    elm.remove();
  })
  return await new Promise((o,x)=>{
    ok = o;
    cancel = x;
  })
}

let $alert = async function(title){
  
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