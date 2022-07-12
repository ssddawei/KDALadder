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
  beginTime;
  endTime;
  personGroup = [];
  constructor(beginTime, personGroup) {
    this.beginTime = beginTime || Date.now();
    this.personGroup = personGroup || [];
  }
  end(endTime) {
    this.endTime = endTime;
  }
}

class Storage {
  ladder = [];
  matches = [];
  load(){};
  save(){};
}