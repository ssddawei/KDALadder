class ALG {
  static PersonScore(kda) {
    return (kda.kill + kda.assist/4) / (kda.death||0.9) * 0.7 + kda.win/(kda.loss||0.9) * 0.3;
  }
}