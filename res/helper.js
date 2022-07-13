let $sel = document.querySelector.bind(document);
let $sels = document.querySelectorAll.bind(document);
let $timeString = (d) => {
  return `${d.getMonth()+1}-${d.getDate()} ${d.getHours()}:${d.getMinutes()}`;
}
let $kdaString = (kda, long) => {
  if(long)
    return `${kda.score.toFixed(1)} ${kda.kill}/${kda.death}/${kda.assist} ${(kda.win/(kda.win+kda.loss)*100).toFixed(1)}%(${kda.win}/${kda.loss})`
  else
  return `${kda.kill}/${kda.death}/${kda.assist}`
}
let $fetch = async function(){
  let response = await fetch.apply(window, arguments)
  if(response.status == 404){
    return null;
  }
  return await response.json();
}