
class ALG {
  static PersonScore(kda) {
    return (kda.kill + kda.assist/4) / (kda.death||0.9) * 0.7 + kda.win/(kda.loss||0.9) * 0.3;
  }
  
  static Sum = arr => arr.reduce((partialSum, a) => partialSum + a, 0)
  static Mean = arr => ALG.Sum(arr) / arr.length;
  static SD(arr) {
    const variance = arr => {
      const m = ALG.Mean(arr);
      return ALG.Sum(arr.map(v => (v - m) ** 2));
    };
    const sd = arr => Math.sqrt(variance(arr) * 1 / (arr.length - 1));
    return sd(arr);
  }

  // 团体的质量：0-1，数字越高质量越高，代表团体成员水平越接近，那么 KDA 得分越准确
  static GroupQuality(ladder) {

    let sdarr = []

    // 胜率代表个体水平，场次越多权重越大
    ladder.forEach(i=>{
      let rate = i.win/(i.win+i.loss) // 胜率
      Array.from({length:i.win+i.loss}).forEach(()=>{
        sdarr.push(rate) // 场次越多，权重越大
      })
    })

    // 胜率的权重标准差
    let sd = ALG.SD(sdarr);

    // 标准差 0.5 表示差距极大，0 表示没差距
    // 归一化
    return Math.max(0, 1 - sd / 0.5)
  }
}