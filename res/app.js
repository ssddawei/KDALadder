import { Match, GameScore } from "./storage.js";
import { LocalStorage } from "./storage-localstorage.js";
import { ALG } from "./algorithm.js";
import { ConnectWebsocket } from "./connect-ws.js";
import { ServerSyncData } from "./sync-server.js";
import "./helper.js";

export class KDAEventCalc {
  // killing-spree  3kill
  // rampage        4kill
  // unstoppable    5kill
  // godlike        6kill
  // legendary      7kill
  // static Legendary = [null, null, null, "killing-spree", "rampage", "unstoppable", "godlike", "legendary"];
  // static Pentakill = [null, null, "2sha", "3sha", "4sha", "5sha"];
  // static Zisha = [null, null, null, "zisha"];
  // static NameToCN = {
  //     "killing-spree": "三杀",
  //     "rampage": "四杀",
  //     "unstoppable": "五杀",
  //     "godlike": "六杀",
  //     "legendary": "七杀",
  //     "2sha": "连续二杀",
  //     "3sha": "连续三杀",
  //     "4sha": "连续四杀",
  //     "5sha": "连续五杀",
  //     "zisha": "自杀",
  //     "shutdown": "终结连杀",
  //     "first-blood": "第一滴血",
  // }
  // static NameToEN = {
  //     "killing-spree": "KillingSpree",
  //     "rampage": "Rampage",
  //     "unstoppable": "Unstoppable",
  //     "godlike": "Godlike",
  //     "legendary": "Legendary",
  //     "2sha": "DoubleKill",
  //     "3sha": "TripleKill",
  //     "4sha": "QuadraKill",
  //     "5sha": "PentaKill",
  //     "zisha": "Executed",
  //     "shutdown": "Shutdown",
  //     "first-blood": "FirstBlood",
  // }

  // info_pentakill = [{}];
  // info_legendary = [{}];
  // info_zisha = [{}];
  // events = [];
  // info_firstBlood = [];

  constructor() {
      this.info_pentakill = [{}];
      this.info_legendary = [{}];
      this.info_zisha = [{}];
      this.events = [];
      this.info_firstBlood = [];
  }

  evolve(score, dryrun) {

    let iEvents = [];
    let curPentakill = this.info_pentakill[this.info_pentakill.length - 1];
    let curLegendary = this.info_legendary[this.info_legendary.length - 1];
    let curZisha = this.info_zisha[this.info_zisha.length - 1];
    let nextPentakill = {}; // recalc pentakill for each score, because pentakill need Continuous Kill
    let nextLegendary = {...curLegendary};
    let nextZisha = {}; // need Continuous Death
    let nextFirstBlood;
    
    

    if(score.death) {
      let person = score.death;
      // firstbood
      if(!this.info_firstBlood.find(i => i)) {
        nextFirstBlood = person;
        iEvents.push({ person, name: "first-blood" });
      }
      // shutdown
      if(nextLegendary[person] >= 5){
        iEvents.push({ person, name: "shutdown" });
      }
      // clear legendary for death
      nextLegendary[person] = 0;
      // zisha
      nextZisha[person] = curZisha[person]? curZisha[person]+1: 1;
      let zisha = KDAEventCalc.Zisha[Math.min(KDAEventCalc.Zisha.length-1, nextZisha[person])];
      if(zisha) {
        iEvents.push({ person, name: zisha });
      }
    } 
    if(score.kill) {
      let person = score.kill;
      nextLegendary[person] = curLegendary[person]? curLegendary[person]+1: 1;
      nextPentakill[person] = curPentakill[person]? curPentakill[person]+1: 1;
      
      let pentakill = KDAEventCalc.Pentakill[nextPentakill[person]];
      let legendary = KDAEventCalc.Legendary[Math.min(KDAEventCalc.Legendary.length-1, nextLegendary[person])];
      if(pentakill) {
        iEvents.push({ person, name: pentakill });
      } 
      if(legendary) {
        iEvents.push({ person, name: legendary });
      }
    }

    if(!dryrun) {
      this.info_pentakill.push(nextPentakill);
      this.info_legendary.push(nextLegendary);
      this.info_zisha.push(nextZisha);
      this.info_firstBlood.push(nextFirstBlood);
      this.events.push(iEvents);
    }
    return iEvents;
  }
  revert() {
    this.info_pentakill.splice(-1, 1);
    this.info_legendary.splice(-1, 1);
    this.info_zisha.splice(-1, 1);
    this.info_firstBlood.splice(-1, 1);

    this.events.splice(-1, 1);
  }
  get currentEvent() {
    return this.events[this.events.length -1] || [];
  }
  static __unittest() {
    let test = new KDAEventCalc();
    test.evolve(new GameScore());
    test.currentEvent.length == 0 || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent.length == 0  || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "2sha" || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "3sha"
      && test.currentEvent[1].name == "killing-spree"  || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "4sha"
      && test.currentEvent[1].name == "rampage"  || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "5sha"
      && test.currentEvent[1].name == "unstoppable"  || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "godlike"  || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "legendary"  || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "legendary"  || console.error("failed");

    // shutdown
    test.evolve(new GameScore(null, "person1"));
    test.currentEvent[1].name == "shutdown"  || console.error("failed");

    // no shutdown
    test.evolve(new GameScore("person1"));
    test.currentEvent.length == 0  || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "2sha" || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "3sha"
      && test.currentEvent[1].name == "killing-spree"  || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "4sha"
      && test.currentEvent[1].name == "rampage"  || console.error("failed");
    test.evolve(new GameScore(null, "person1"));
    test.currentEvent.length == 0  || console.error("failed");

    // godlike and double-kill
    test.evolve(new GameScore("person1"));
    test.currentEvent.length == 0  || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "2sha" || console.error("failed");
    test.evolve(new GameScore("person2")); // stop 3sha
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "killing-spree"  || console.error("failed");
    test.evolve(new GameScore("person1"));
    test.currentEvent[0].name == "2sha"
      && test.currentEvent[1].name == "rampage"  || console.error("failed");
    test.evolve(new GameScore("person1")); // godlike
    test.evolve(new GameScore(null, "person1"));
    test.currentEvent[0].name == "shutdown"  || console.error("failed");

    // first-blood
    let testFB = new KDAEventCalc();
    testFB.evolve(new GameScore("person1"));
    testFB.evolve(new GameScore("person2"));
    testFB.evolve(new GameScore(null, "person1"));
    testFB.currentEvent[0].name == "first-blood"  || console.error("failed");
    testFB.evolve(new GameScore(null, "person1"));
    testFB.currentEvent.length == 0  || console.error("failed");
  }
}
 
KDAEventCalc.Legendary = [null, null, null, "killing-spree", "rampage", "unstoppable", "godlike", "legendary"];
KDAEventCalc.Pentakill = [null, null, "2sha", "3sha", "4sha", "5sha"];
KDAEventCalc.Zisha = [null, null, null, "zisha"];
KDAEventCalc.NameToCN = {
    "killing-spree": "三杀",
    "rampage": "四杀",
    "unstoppable": "五杀",
    "godlike": "六杀",
    "legendary": "七杀",
    "2sha": "连续二杀",
    "3sha": "连续三杀",
    "4sha": "连续四杀",
    "5sha": "连续五杀",
    "zisha": "自杀",
    "shutdown": "终结连杀",
    "first-blood": "第一滴血",
}
KDAEventCalc.NameToEN = {
    "killing-spree": "KillingSpree",
    "rampage": "Rampage",
    "unstoppable": "Unstoppable",
    "godlike": "Godlike",
    "legendary": "Legendary",
    "2sha": "DoubleKill",
    "3sha": "TripleKill",
    "4sha": "QuadraKill",
    "5sha": "PentaKill",
    "zisha": "Executed",
    "shutdown": "Shutdown",
    "first-blood": "FirstBlood",
}

export class GroupController {
  constructor() {
    this.sync = new ServerSyncData()
  }
  get online() {
    return this.sync.online;
  }
  async register(groupCode, groupName, inviteCode) {
    let result = await this.sync.register(groupCode, groupName, inviteCode)
  }
  async login(groupCode) {
    let result = await this.sync.login(groupCode);

  }
  async logout() {
    this.sync.logout()
  }
  async info() {
    return await this.sync.info();
  }
  async updateInfo(groupName, groupCode) {
    let result = await this.sync.updateInfo(groupName, groupCode);
    if(groupCode) {
      await this.login(groupCode);
    }
  }
}

export class MatchController {
  // _match = new Match();
  // eventCalc = new KDAEventCalc();
  // eventCallback;
  get match() {
    return this._match;
  }
  set match(m) { // recalc event in every match setted
    this._match = m;
    this.eventCalc = new KDAEventCalc();
    m.scores.forEach(i => this.eventCalc.evolve(i));
    let event = this.eventCalc.currentEvent;
    event.length && this.onEvent(event)
    if(this.started && !this.ended)
      SoundEffect.speak(`比分 ${this.scoreText}`);
  }
  get ready() {
    return !!(this.match.personGroup.filter(i=>i).length == 4)
  }
  get started() {
    return this.aScore > 0 || this.bScore > 0;
  }
  get readyToEnd() {
    return Math.abs(this.aScore - this.bScore) >= 2 && (this.aScore > 20 || this.bScore > 20)
  }
  get ended() {
    return !!(this.match.scores.filter(i=>i.win || i.loss).length)
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
  get scoreText() {
    let aFirst = this.match.scores.slice(-1).filter(
      i => this.aGroup.indexOf(i.kill) >= 0 || this.bGroup.indexOf(i.death) >= 0
    ).length > 0;
    let a = this.aScore % 10;
    let b = this.bScore % 10;
    return aFirst? `${a} ${b}`: `${b} ${a}`;
  }
  constructor(aGroupOrMatch = [], bGroup = []) {
    this._match = new Match();
    this.eventCalc = new KDAEventCalc();
    // eventCallback;
    if(aGroupOrMatch instanceof Match || aGroupOrMatch.scores) {
      this.match = aGroupOrMatch;
    } else {
      this.match.personGroup = [...aGroupOrMatch, ...bGroup];
    }
  }
  goal(person) {
    let assistGroup = this.aGroup.indexOf(person) >= 0? this.aGroup: this.bGroup;
    let assist = assistGroup.filter(i => i != person)[0];
    this.match.scores.push(new GameScore(person, null, assist));
    if(this.match.scores.length == 1) {
      this.match.beginTime = Date.now()
    }
    let event = this.eventCalc.evolve(this.match.scores[this.match.scores.length - 1]);
    event.length && this.onEvent(event);
    SoundEffect.speak(`比分 ${this.scoreText}`);
  }
  loss(person) {
    this.match.scores.push(new GameScore(null, person))
    if(this.match.scores.length == 1) {
      this.match.beginTime = Date.now()
    }
    let event = this.eventCalc.evolve(this.match.scores[this.match.scores.length - 1]);
    event.length && this.onEvent(event);
    SoundEffect.speak(`比分 ${this.scoreText}`);
  }
  revert() {
    this.match.scores.length && this.match.scores.length --;
    this.eventCalc.revert();
  }
  kda(person) {
    if(+person < 4) {
      person = this.match.personGroup[+person];
    }
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
  mvp() {
    if(!this.started)return;
    let kda = this.match.personGroup.map(this.kda.bind(this));
    if(this.match.scores.filter(i => i.win || i.loss).length <= 0) {
      if(this.aScore > this.bScore) {
        kda[0].win = kda[1].win = 1;
      } else {
        kda[2].win = kda[3].win = 1;
      }
    }
    let mvp = this.match.personGroup[Object.entries(kda).sort((a,b)=>b[1].score-a[1].score)[0][0]];
    if(this._mvp != mvp) {
      this.onEvent([{person: mvp, name: "mvp_changed"}])
    }
    return this._mvp = mvp;
  }
  loser() {
    if(!this.started)return;
    let kda = this.match.personGroup.map(this.kda.bind(this));
    if(this.match.scores.filter(i => i.win || i.loss).length <= 0) {
      if(this.aScore > this.bScore) {
        kda[0].win = kda[1].win = 1;
      } else {
        kda[2].win = kda[3].win = 1;
      }
    }
    let loser = this.match.personGroup[Object.entries(kda).sort((a,b)=>a[1].score-b[1].score)[0][0]];
    if(this._loser != loser) {
      this.onEvent([{person: loser, name: "loser_changed"}])
    }
    return this._loser = loser;
  }
  nextEvent(person, killOrDeath) {
    return this.eventCalc.evolve(new GameScore(killOrDeath && person, !killOrDeath && person), true)
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
    storage.data = {"current": this.match}
    storage.save();
  }
  load() {
    let storage = new LocalStorage("current");
    if(storage.data.current)
      this.match = storage.data.current;
  }
  ladder() {
    let ladder = new Ladder();
    ladder.beginTime = this.match.beginTime;
    ladder.endTime = this.match.endTime2;
    ladder.matchCount = 1;
    ladder.matchTotalTimeSec = Math.floor((this.match.endTime2 - this.match.beginTime)/1000);
    
    // update to local ladder

    let kdas = this.match.personGroup.map(i=>this.kda(i));

    MatchController.LadderEvolve(ladder.ladder, this.aGroup[0], kdas[0]);
    MatchController.LadderEvolve(ladder.ladder, this.aGroup[1], kdas[1]);
    MatchController.LadderEvolve(ladder.ladder, this.bGroup[0], kdas[2]);
    MatchController.LadderEvolve(ladder.ladder, this.bGroup[1], kdas[3]);
    
    return ladder;
  }
  onEvent(e) {
    (async ()=>{
      for(let i in e) {
        // SoundEffect.play(e[i].name);
        if(e[i].name == "mvp_changed") {
          SoundEffect.speak("mvp 榜主......" + e[i].person);
        } else if (e[i].name == "loser_changed") {
          SoundEffect.speak("loser 榜主......" + e[i].person);
        }
        i < e.length - 1 && await new Promise(o=>setTimeout(o, 2000));
      }
    })()
    let outEvents = e.filter(i=>i.name != "text")
    this.eventCallback && outEvents.length && this.eventCallback(outEvents);
  }
  async end() {

    if(!this.ready || !this.started) return;

    let storage = new LocalStorage();
    let sync = new ServerSyncData(storage, new LocalStorage("remote"));

    let now = new Date();

    // add win/loss score
    if( this.aScore > this.bScore ) {
      this.match.scores.push(new MatchScore(this.aGroup, this.bGroup))
    } else {
      this.match.scores.push(new MatchScore(this.bGroup, this.aGroup))
    }
    
    // save to local
    (storage.data[$dateString(now)] = storage.data[$dateString(now)] || [])
      .push(this.match);

    (storage.ladder[$seasonString(now)] = storage.ladder[$seasonString(now)] || [])
      .push(this.ladder());

    storage.save();

    // sync
    await sync.sync();

    new LocalStorage("current").delete();

    this.match = new Match();

    return $dateString(now);
  }
}

export class LadderController {
  // syncData = new ServerSyncData(null, new LocalStorage("remote"));
  constructor() {
    this.syncData = new ServerSyncData(null, new LocalStorage("remote"));
  }
  get storage() {
    return this.syncData.local
  }
  get remote() {
    return this.syncData.remoteCache;
  }
  async seasonLadder(season = $seasonString(new Date())) {
    let ladder = this.remote.ladder[season];
    if(true || !ladder) {
      let dateOfSession = season
        .replace("season1", "1-1")
        .replace("season2", "4-1")
        .replace("season3", "7-1")
        .replace("season4", "10-1") 
      await this.syncData.loadRemote(new Date(dateOfSession));
      ladder = this.remote.ladder[season];
    }
    
    // sum each date' ladders of this season
    return ladder.reduce((pre, nxt) => {
      nxt.ladder.forEach(kda => {
        MatchController.LadderEvolve(pre.ladder, kda.person, kda);
      })
      return pre;
    }, {ladder:[]})
  }
  async seasonHallLadder(seasonDate = new Date()) {
    let [allLadders, allName] = await this.syncData.loadHallLadder(seasonDate);

    // calc each ladder
    allLadders = allLadders.map(ladder => ladder.reduce((pre, nxt) => {
      nxt.ladder.forEach(kda => {
        MatchController.LadderEvolve(pre.ladder, kda.person, kda);
      })
      return pre;
    }, {ladder:[]}) )

    allLadders.forEach((i,idx) => {
      i.name = allName[idx]
      i.quality = ALG.GroupQuality(i.ladder)
    })

    // calc hall ladder
    allLadders = allLadders.reduce((pre, nxt) => {
      nxt.ladder.forEach(kda => {
        kda.group = nxt.name;
        kda.quality = nxt.quality;
        kda.score *= nxt.quality;
        if(kda.win + kda.loss > 5)
          pre.ladder.push(kda);
        // MatchController.LadderEvolve(pre.ladder, kda.person, kda);
      })
      return pre;
    }, {ladder:[]})

    allLadders.ladder.sort((a,b)=>b.score-a.score)

    return allLadders;
  }
  async dateLadder(date = $dateString(new Date()), beginTime) {
    let season = $seasonString(new Date(date));
    let ladder = this.remote.ladder[season];
    
    if(!ladder || (beginTime && !ladder.filter(i=>i.beginTime == beginTime).length)) {
      await this.syncData.loadRemote(new Date(date));
      ladder = this.remote.ladder[season];
    }
    
    // not found in REMOTE , and found match , calc ladder
    if(!ladder.length && this.dateMatch(date)) {
      let matches = await this.dateMatch(date);
      ladder =  matches.map(match => {
        return new MatchController(match).ladder()
      })

      // cache
      this.syncData.remoteCache.ladder[season] = ladder;
      this.syncData.remoteCache.save();
    }

    // sum each date' ladders of this season
    return ladder.reduce((pre, nxt) => {
      if($dateString(new Date(nxt.beginTime)) == date)
        nxt.ladder.forEach(kda => {
          MatchController.LadderEvolve(pre.ladder, kda.person, kda);
        })
      return pre;
    }, {ladder:[]})
  }
  async dateMatch(date = $dateString(new Date()), beginTime) {
    let match = this.remote.data[date];
    if(!match || !match.length || (beginTime && !match.filter(i=>i.beginTime == beginTime).length)) {
      await this.syncData.loadRemote(new Date(date));
    }
    return this.remote.data[date];
  }

  async sync() {
    await this.syncData.loadRemote(new Date());
  }
}

/*
  Bind ConnectWebrtc to UI
*/
export class ConnectController {
  // conn;
  // status;
  // mode;
  // onData;
  constructor(mode, onData) {
    this.mode = mode; // act as "server" or "client"
    this.onData = onData; // on data receive

    $sel(".connect").addEventListener("click", async () => {

      if(this.status == "done" ||this.status == "loading"){
        // if(await $confirm("确定断开？")){
          this.conn && this.conn.close()
          this.conn == undefined;
          localStorage.removeItem("connect-status");
          this.refreshUI();
          return;
        // }
      }

      this.connect();
    });

    if(localStorage.getItem("connect-status") == "done") {
      this.connect();
    }
  }
  set subgroup(val) {
    let needSendSetSubGroup = this._subgroup != val;
      
    this._subgroup = val;
    localStorage.setItem("__subgroup", val);

    needSendSetSubGroup && this.send("set-subgroup")
  }
  get subgroup() {
    if(!this._subgroup) {
      this._subgroup = localStorage.getItem("__subgroup") || 1;
    }
    return this._subgroup;
  }
  refreshUI(status) {
    this.status = status;

    $sel(".connect").classList.remove("loading");
    $sel(".connect").classList.remove("error");
    $sel(".connect").classList.remove("done");
    if(this.status && typeof(this.status) == "string")
      $sel(".connect").classList.add(this.status);
    else if(this.status && typeof(this.status) == "object")
      $sel(".connect").classList.add(...this.status);

  }
  async connect() {

    localStorage.setItem("connect-status", "done");
    this.refreshUI("loading");

    if(this.conn){ 
      this.conn.close();
      this.conn = undefined;
    }

    // create ConnectWebrtc, set receiveCallback, errorCallback
    this.conn = new ConnectWebsocket(ServerSyncData.key.groupCode, 
      (msg) => {
        if(msg.indexOf("hi") == 0) {
          this.refreshUI("done");
          this.onData && this.onData({action: "connect"});
          this.send("set-subgroup")
          return;
        }
        
        let data = JSON.parse(msg);
        this.onData && this.onData(data);
      }, (err) => {
        if(err.code && err.code != 1000) {
          this.refreshUI(["error", "done"]);
          setTimeout(()=>{this.connect()}, 3000); // retry
        } else {
          this.refreshUI();
        }
      });


  }
  send(action, data) {
    this.status == "done" && this.conn.send(JSON.stringify({action, data, subgroup: this.subgroup}));
  }
}
export class Menu {
  constructor() {
    $sel(".menuBtn").addEventListener("click", ()=>{
      $sel(".menu").classList.add("show");
    })
    $sel(".menu").addEventListener("click", ()=>{
      $sel(".menu").classList.remove("show");
    })
    let btn = $sel(".menuBtn");
    let menu = $sel(".menu");
    document.addEventListener("click", function(e){
      if(e.composedPath().filter(i=>i==btn||i==menu).length == 0){
        $sel(".menu").classList.remove("show");
      }
    })
  }
}

export class ListChooser {
  // chooseCallback;
  // loader;
  constructor(loader) {
    this.loader = loader;
    $sel("div.dataList .list").addEventListener("click", (e) => {
      let data = e.composedPath().filter(i => i.dataset && i.dataset["data"])[0];
      this.select(data && data.dataset["data"]);
    })
    $sel("div.dataList .cancelBtn").addEventListener("click", () => {
      this.cancel();
    });
    $sel("div.dataList").addEventListener("click", (e) => {
      if(e.target == $sel("div.dataList"))
        this.cancel();
    });
  }
  async refreshUI() {
    let datas = this.loader && await this.loader() || [];
    let tpl = $sel("#DataListItem").innerHTML;
    $sel("div.dataList > .list").innerHTML = datas.map(d => {
      let data = d.data || d;
      let subtitle = d.subtitle || "";
      let active = d.active? "active": "";
      return tpl.replace(/{{data}}/g, data)
        .replace("{{active}}", active)
        .replace("{{subtitle}}", subtitle);
    }).join("");
  }
  select(date) {
    this.chooseCallback && this.chooseCallback(date);
    $sel(".dataList").classList.remove("show")
    $popHistoryBack();
  }
  cancel() {
    this.chooseCallback && this.chooseCallback();
    $sel(".dataList").classList.remove("show")
    $popHistoryBack();
  }
  choose() {
    this.refreshUI();
    $sel(".dataList").classList.add("show")

    $pushHistoryBack(this.cancel.bind(this))
    
    return new Promise(done => {
      this.chooseCallback = done;
    })
  };
}

export class PlaceHolder {
  constructor() {
    $sels(".place-holder-owner").forEach(i => {
      i.addEventListener("click", () => {
        let next = i.nextElementSibling;
        if(next.classList.contains("place-holder")) {
          next.classList.add("show");
          i.classList.add("hide");
        }
      })
    })
    $sels(".place-holder").forEach(i => {
      i.addEventListener("click", () => {
        let prev = i.previousElementSibling;
        if(prev.classList.contains("place-holder-owner")) {
          prev.classList.remove("hide");
          i.classList.remove("show");
        }
      })
    })
  }
}


export class SoundEffect {
  // static audio = [];
  static get disabled() {
    return !!localStorage.getItem("music-disabled");
  }
  static set disabled(d) {
    if(d)
      localStorage.setItem("music-disabled", true);
    else
      localStorage.removeItem("music-disabled");
    SoundEffect.refreshUI();
  }
  static speak(text) {
    if(SoundEffect.disabled)return;
    if(!window.SpeechSynthesisUtterance)return;
    
    let msg = new SpeechSynthesisUtterance();
    msg.text = text;
    msg.rate = 0.7;
    msg.pitch = 1;
    msg.lang = "zh-HK";
    speechSynthesis.speak(msg);
  }
  static play(effect, audioSeq = 0) {
    if(SoundEffect.disabled)return;

    if(!SoundEffect.audio[audioSeq]){
      SoundEffect.audio[audioSeq] = new Audio();
    }
    SoundEffect.audio[audioSeq].src = "res/sound/a-" + effect + ".ogg";
    SoundEffect.audio[audioSeq].play();
  }
  static bindUI() {
    if(!$sel(".musicBtn"))return;
    SoundEffect.refreshUI();
    $sel(".musicBtn").addEventListener("click", () => {
      SoundEffect.disabled = !SoundEffect.disabled;
      SoundEffect.refreshUI();
    })
  }
  static refreshUI() {
    if(!$sel(".musicBtn"))return;
    if(SoundEffect.disabled) {
      $sel(".musicBtn").classList.add("disabled");
    } else {
      $sel(".musicBtn").classList.remove("disabled");
    }
  }
}
SoundEffect.audio = [];

window.addEventListener("load", ()=>{
  SoundEffect.bindUI();
  new PlaceHolder();
})