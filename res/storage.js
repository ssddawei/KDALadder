class Score {
  timestamp;
  constructor(TS) {
    this.timestamp = TS || Date.now();
  }
}

class GameScore extends Score {
  kill;
  death;
  assist;
  constructor(K, D, A, TS) {
    super(TS)
    this.kill = K;
    this.death = D;
    this.assist = A;
  }
}

class MatchScore extends Score {
  win = [];
  loss = [];
  constructor(W, L, TS) {
    super(TS)
    this.win = W;
    this.loss = L;
  }
}

class Match {
  scores = [];
  beginTime = 0;
  endTime = 0;
  personGroup = [];
  get endTime2() {
    if(this.endTime) {
      return this.endTime;
    } else {
      let lastScore = this.scores.slice(-1)[0];
      return lastScore && lastScore.timestamp; 
    }
  }
  constructor(beginTime, personGroup) {
    this.beginTime = beginTime || Date.now();
    this.personGroup = personGroup || [];
  }
  end(endTime) {
    this.endTime = endTime;
  }
}

class Ladder {
  beginTime = 0;
  endTime = 0;
  matchCount = 0;
  matchTotalTimeSec = 0;
  ladder = [];
}

class Storage {
  index = ALG.StorageIndex(); 
  ladder = new Ladder();
  matches = [];
  constructor(index) {
    if(index != undefined)
      this.index = index;
  }
  load(){};
  save(){};
  delete(){};
}