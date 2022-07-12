# 目录结构
/ 
  index.html
  match.html
  ladder.html
  score.html
  /res/
    app.js
    app.css
  /data/
    ladder.json
    data-2022-07.json

# 数据结构 
ladder
[
  {
    person: someone,
    K: number,
    D: number,
    A: number,
    win: number,
    loss: number,
    score: number,
  }
]

data
{
  matches: [
    {
      beginTime: datetime,
      endTime: datetime,
      scores: [
        {
          ts: datetime,
          K: someone,
          D: someone,
          A: someone,
        },
        {
          ts: datetime,
          win: [someone],
          loss: [someone],
        }
      ]
    }
  ]
}