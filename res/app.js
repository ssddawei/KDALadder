
class MatchController {
  match = new Match();
  get ready() {
    return !!(this.match.personGroup.filter(i=>i).length == 4)
  }
  get aGroup() {
    return this.match.personGroup.slice(0,2);
  }
  get bGroup() {
    return this.match.personGroup.slice(2,4);
  }
  get aScore() {
    return this.match.scores.filter(i => this.aGroup.indexOf(i.kill) >= 0).length +
      this.match.scores.filter(i => this.bGroup.indexOf(i.death) >= 0).length
  }
  get bScore() {
    return this.match.scores.filter(i => this.bGroup.indexOf(i.kill) >= 0).length +
      this.match.scores.filter(i => this.aGroup.indexOf(i.death) >= 0).length
  }
  constructor(aGroupOrMatch = [], bGroup = []) {
    if(aGroupOrMatch instanceof Match || aGroupOrMatch.scores) {
      this.match = aGroupOrMatch;
    } else {
      this.match.personGroup = [...aGroupOrMatch, ...bGroup];
    }
  }
  goal(person) {
    let assistGroup = this.aGroup.indexOf(person) >= 0? this.aGroup: this.bGroup;
    let assist = assistGroup.filter(i => i != person)[0];
    this.match.scores.push(new GameScore(person, null, assist))
  }
  loss(person) {
    this.match.scores.push(new GameScore(null, person))
  }
  revert() {
    this.match.scores.length && this.match.scores.length --;
  }
  kda(person) {
    let kill = this.match.scores.filter(i => i.kill == person).length;
    let death = this.match.scores.filter(i => i.death == person).length;
    let assist = this.match.scores.filter(i => i.assist == person).length;
    let win = this.match.scores.filter(i => i.win && i.win.indexOf(person) >= 0).length;
    let loss = this.match.scores.filter(i => i.loss && i.loss.indexOf(person) >= 0).length;
    let kda = {
      kill, death, assist, win, loss
    }
    kda.score = ALG.PersonScore(kda);
    return kda;
  }
  static LadderEvolve(ladder, person, kda) {
    let item = ladder.filter(i => i.person == person)[0];
    if(!item) {
      ladder.push(item = { 
        person
      });
    }
    item.kill = (+item.kill || 0) + (+kda.kill || 0);
    item.death = (+item.death || 0) + (+kda.death || 0);
    item.assist = (+item.assist || 0) + (+kda.assist || 0);
    item.win = (+item.win || 0) + (+kda.win || 0);
    item.loss = (+item.loss || 0) + (+kda.loss || 0);
    item.score = ALG.PersonScore(item);
    ladder.sort((a,b) => b.score - a.score)
  }
  save() {
    let storage = new LocalStorage("current");
    storage.matches[0] = (this.match);
    storage.save();
  }
  load() {
    let storage = new LocalStorage("current");
    storage.load();
    if(storage.matches.length) {
      this.match = storage.matches[0];
    }
  }
  end() {

    if(!this.ready) return;

    let storage = new LocalStorage();
    let sync = new AliyunSyncData(storage);

    // check sync key
    if(!sync.key) {
      let key = prompt("同步到云端，请输入密钥");
      if(!key) {
        if(!confirm("不保存到云端，确认？")){
          return;
        }
      } else {
        if(!sync.saveKey(key)){
          alert("密钥格式错误");
          return;
        }
      }
    }

    // add win/loss score
    if( this.aScore > this.bScore ) {
      this.match.scores.push(new MatchScore(this.aGroup, this.bGroup))
    } else {
      this.match.scores.push(new MatchScore(this.bGroup, this.aGroup))
    }
    
    // save to local
    storage.matches.push(this.match);

    storage.ladder.beginTime = storage.matches[0].beginTime;
    storage.ladder.endTime = this.match.endTime2;
    storage.ladder.matchCount = (+storage.ladder.matchCount||0) + 1;
    storage.ladder.matchTotalTimeSec = (+storage.ladder.matchTotalTimeSec||0) + ((this.match.endTime2 - this.match.beginTime)/1000).toFixed(0);
    
    // update to local ladder
    MatchController.LadderEvolve(storage.ladder.ladder, this.aGroup[0], this.kda(this.aGroup[0]));
    MatchController.LadderEvolve(storage.ladder.ladder, this.aGroup[1], this.kda(this.aGroup[1]));
    MatchController.LadderEvolve(storage.ladder.ladder, this.bGroup[0], this.kda(this.bGroup[0]));
    MatchController.LadderEvolve(storage.ladder.ladder, this.bGroup[1], this.kda(this.bGroup[1]));

    storage.save();

    // sync
    sync.sync();

    new LocalStorage("current").delete();

    this.match = new Match();

    return storage.index;
  }
}
