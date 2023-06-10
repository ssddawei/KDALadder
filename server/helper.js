
let timeString = (d) => {
  return `${d.getMonth()+1}-${d.getDate()} ${d.getHours()}:${d.getMinutes()}`;
}
let dateString = (d) => {
  return `${d.getFullYear()}-${(d.getMonth()+1).toString().padStart(2, '0')}-${d.getDate().toString().padStart(2, '0')}`;
}
let tickString = (d) => {
  return dateString(d) + "_" + `${d.getHours()}-${d.getMinutes()}-${d.getSeconds()}-${d.getMilliseconds()}`;
}
let seasonString = (d) => {
  switch(d.getMonth()) {
    case 0:case 1: case 2: return `${d.getFullYear()}-season1`;
    case 3:case 4: case 5: return `${d.getFullYear()}-season2`;
    case 6:case 7: case 8: return `${d.getFullYear()}-season3`;
    case 9:case 10: case 11: return `${d.getFullYear()}-season4`;
  }
}

export {
  timeString, dateString, seasonString, tickString
}
