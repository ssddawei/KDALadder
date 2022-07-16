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
数据使用半json格式，只能添加不能修改。获取到数据后，用[]包裹后解释。
## 季度天梯数据，1-3 4-6 7-9 10-12 分别记录
每场比赛叠加数据进去
```
ladder-2022-season1.json
{
  beginTime: datetime,
  endTime: datetime,
  matchCount: number,
  matchTotalTimeSec: number,
  ladder: [
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
},
```

## 比赛记录
每场比赛叠加数据进去
```
data-2022-07-21
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
},
```