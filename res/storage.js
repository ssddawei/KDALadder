export class Score {
  // timestamp;
  constructor(TS) {
    this.timestamp = TS || Date.now();
  }
}

export class GameScore extends Score {
  // kill;
  // death;
  // assist;
  constructor(K, D, A, TS) {
    super(TS)
    this.kill = K;
    this.death = D;
    this.assist = A;
  }
}

export class MatchScore extends Score {
  // win = [];
  // loss = [];
  constructor(W, L, TS) {
    super(TS)
    this.win = W;
    this.loss = L;
  }
}

export class Match {
  // scores = [];
  // beginTime = 0;
  // endTime = 0;
  // personGroup = [];
  get endTime2() {
    if(this.endTime) {
      return this.endTime;
    } else {
      let lastScore = this.scores.slice(-1)[0];
      return lastScore && lastScore.timestamp; 
    }
  }
  constructor(beginTime, personGroup) {
    this.scores = [];
    this.beginTime = 0;
    this.endTime = 0;
    this.personGroup = [];

    this.beginTime = beginTime || Date.now();
    this.personGroup = personGroup || [];
  }
  end(endTime) {
    this.endTime = endTime;
  }
}

export class Ladder {
  // beginTime = 0;
  // endTime = 0;
  // matchCount = 0;
  // matchTotalTimeSec = 0;
  // ladder = [];
  constructor() {
    this.beginTime = 0;
    this.endTime = 0;
    this.matchCount = 0;
    this.matchTotalTimeSec = 0;
    this.ladder = [];
  }
}

export class Storage {
  // prefix = "";
  // ladder = {};
  // data = {};
  constructor(prefix) {
    this.prefix = (prefix || "");
    this.ladder = {};
    this.data = {};
  }
  load(){};
  save(){};
  delete(){};
}