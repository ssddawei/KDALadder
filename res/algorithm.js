class ALG {
  static PersonScore(kda) {
    return (kda.kill + kda.assist/2) / (kda.death||0.5) * 0.5 + kda.win/(kda.loss||0.5) * 0.5;
  }
  static StorageIndex(date) {
    if(!date) {
      date = new Date();
    }
    return `${date.getFullYear()}-${date.getMonth()+1}-${date.getDate()}`;
  }
}