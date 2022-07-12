
class MatchController {
  match = new Match();
  aGroup = [];
  bGroup = [];
  get aScore() {
    return this.match.scores.filter(i => this.aGroup.indexOf(i.kill) >= 0).length +
      this.match.scores.filter(i => this.bGroup.indexOf(i.death) >= 0).length
  }
  get bScore() {
    return this.match.scores.filter(i => this.bGroup.indexOf(i.kill) >= 0).length +
      this.match.scores.filter(i => this.aGroup.indexOf(i.death) >= 0).length
  }
  constructor(aGroupOrMatch, bGroup) {
    if(aGroupOrMatch instanceof Match || aGroupOrMatch.scores) {
      this.match = aGroupOrMatch;
      this.aGroup = this.match.personGroup.slice(0,2);
      this.bGroup = this.match.personGroup.slice(2,4);
    } else {
      this.aGroup = aGroupOrMatch;
      this.bGroup = bGroup;
      this.match.personGroup = [...aGroupOrMatch, ...bGroup];
    }
  }
  start(aGroup, bGroup) {
    this.aGroup = aGroup;
    this.bGroup = bGroup;
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
  end() {

    // add win/loss score
    if( this.aScore > this.bScore ) {
      this.match.scores.push(new MatchScore(this.aGroup, this.bGroup))
    } else {
      this.match.scores.push(new MatchScore(this.bGroup, this.aGroup))
    }
    
    // save to local
    let storage = new LocalStorage();
    storage.matches.push(this.match);

    // update to local ladder
    let updateLadderPerson = (person) => {
      let kda = this.kda(person);
      let item = storage.ladder.filter(i => i.person == person)[0];
      if(!item) {
        storage.ladder.push(item = { 
          person
        });
      }
      item.kill = (+item.kill || 0) + (+kda.kill || 0);
      item.death = (+item.death || 0) + (+kda.death || 0);
      item.assist = (+item.assist || 0) + (+kda.assist || 0);
      item.win = (+item.win || 0) + (+kda.win || 0);
      item.loss = (+item.loss || 0) + (+kda.loss || 0);
    }
    
    updateLadderPerson(this.aGroup[0])
    updateLadderPerson(this.aGroup[1])
    updateLadderPerson(this.bGroup[0])
    updateLadderPerson(this.bGroup[1])

    storage.ladder.forEach(i => {
      i.score = ALG.PersonScore(i);
    })
    storage.ladder = storage.ladder.sort((a,b) => b.score - a.score)

    storage.save();

    // sync
    new AliyunSyncData(storage).sync();

  }
}
