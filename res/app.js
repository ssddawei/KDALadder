
class KDAEventCalc {
  // killing-spree  3kill
  // rampage        4kill
  // unstoppable    5kill
  // godlike        6kill
  // legendary      7kill
  static Legendary = [null, null, null, "killing-spree", "rampage", "unstoppable", "godlike", "legendary"];
  static Pentakill = [null, null, "2sha", "3sha", "4sha", "5sha"];
  static Zisha = [null, null, null, "zisha"];
  static NameToCN = {
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
  static NameToEN = {
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

  info_pentakill = [{}];
  info_legendary = [{}];
  info_zisha = [{}];
  events = [];
  info_firstBlood = [];

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
    return this.events[this.events.length -1];
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
class MatchController {
  _match = new Match();
  eventCalc = new KDAEventCalc();
  eventCallback;
  get match() {
    return this._match;
  }
  set match(m) { // recalc event in every match setted
    this._match = m;
    this.eventCalc = new KDAEventCalc();
    m.scores.forEach(i => this.eventCalc.evolve(i));
    this.eventCalc.currentEvent && this.onEvent(this.eventCalc.currentEvent)
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
    this.match.scores.push(new GameScore(person, null, assist));
    let event = this.eventCalc.evolve(this.match.scores[this.match.scores.length - 1]);
    event.length && this.onEvent(event);
  }
  loss(person) {
    this.match.scores.push(new GameScore(null, person))
    let event = this.eventCalc.evolve(this.match.scores[this.match.scores.length - 1]);
    event.length && this.onEvent(event);
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
    return this.match.personGroup[Object.entries(kda).sort((a,b)=>b[1].score-a[1].score)[0][0]];
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
    return this.match.personGroup[Object.entries(kda).sort((a,b)=>a[1].score-b[1].score)[0][0]];
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
    MatchController.LadderEvolve(ladder.ladder, this.aGroup[0], this.kda(this.aGroup[0]));
    MatchController.LadderEvolve(ladder.ladder, this.aGroup[1], this.kda(this.aGroup[1]));
    MatchController.LadderEvolve(ladder.ladder, this.bGroup[0], this.kda(this.bGroup[0]));
    MatchController.LadderEvolve(ladder.ladder, this.bGroup[1], this.kda(this.bGroup[1]));
    
    return ladder;
  }
  onEvent(e) {
    (async ()=>{
      for(let i in e) {
        SoundEffect.play(e[i].name);
        i < e.length - 1 && await new Promise(o=>setTimeout(o, 2000));
      }
    })()
    this.eventCallback && this.eventCallback(e);
  }
  async end() {

    if(!this.ready || !this.started) return;

    let storage = new LocalStorage();
    let sync = new AliyunSyncData(storage, new LocalStorage("remote"));

    // check sync key
    if(!sync.key) {
      let key = await $prompt("同步到云端，请输入密钥");
      if(!key) {
        if(!await $confirm("不保存到云端，确认？")){
          return;
        }
      } else {
        if(!sync.saveKey(key)){
          await $alert("密钥格式错误");
          return;
        }
      }
    }

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

class LadderController {
  syncData = new AliyunSyncData(null, new LocalStorage("remote"));

  get storage() {
    return this.syncData.local
  }
  get remote() {
    return this.syncData.remoteCache;
  }
  async seasonLadder(season = $seasonString(new Date())) {
    let ladder = this.remote.ladder[season];
    if(!ladder) {
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
  async dateLadder(date = $dateString(new Date())) {
    let season = $seasonString(new Date(date));
    let ladder = this.remote.ladder[season];
    
    if(!ladder) {
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
  async dateMatch(date = $dateString(new Date())) {
    let match = this.remote.data[date];
    if(!match) {
      await this.syncData.loadRemote(new Date(date));
    }
    return this.remote.data[date];
  }

  async sync() {
    await this.syncData.sync();
  }
}

/*
  Bind ConnectWebrtc to UI
*/
class ConnectController {
  conn;
  status;
  mode;
  onData;
  constructor(mode, onData) {
    this.mode = mode; // act as "server" or "client"
    this.onData = onData; // on data receive

    $sel(".connect").addEventListener("click", async () => {

      // check sync key
      if(!new AliyunSyncData().key) {
        let key = await $prompt("多端连接，请输入密钥");
        if(!key) {
          return;
        } else {
          if(!new AliyunSyncData().saveKey(key)){
            await $alert("密钥格式错误");
            return;
          }
        }
      }

      this.refreshUI("loading");
      this.connect();
    });
  }
  refreshUI(status) {
    this.status = status;

    $sel(".connect").classList.remove("loading");
    $sel(".connect").classList.remove("error");
    $sel(".connect").classList.remove("done");
    if(typeof(this.status) == "string")
      $sel(".connect").classList.add(this.status);
    else if(typeof(this.status) == "object")
      $sel(".connect").classList.add(...this.status);
  }
  async connect() {
    if(this.conn){ 
      this.conn.close();
      this.conn = undefined;
    }

    // create ConnectWebrtc, set receiveCallback, errorCallback
    let conn = this.conn = new ConnectWebrtc(new AliyunSyncData(), 
      (msg) => {
        this.refreshUI("done");
        this.onData && this.onData({action: "connect"});
        if(msg.indexOf("hi") == 0)return;
        
        let data = JSON.parse(msg);
        this.onData && this.onData(data);
      }, (err) => {
        this.refreshUI(["error", "done"]);
      });

    try{

      // invoke offer/answer to connect Client/Server
      if(this.mode == "server")
        await conn.offer();
      else if(this.mode == "client")
        await conn.answer();
      else
        throw new Error("wrong mode: " + mode)

      conn.send("hi " + this.mode);

    } catch(e) {
      if(conn != this.conn) {
        // conn released
      }
      else if(e && e.message == "timeout") {
        this.refreshUI();
      } else if(e && e.message == "canceled") {
        // conn canceled
      } else {
        $alert("连接失败: " + e)
        this.refreshUI("error");
      }
    }


  }
  send(action, data) {
    this.status == "done" && this.conn.send(JSON.stringify({action, data}));
  }
}
class Menu {
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
      if(e.path.filter(i=>i==btn||i==menu).length == 0){
        $sel(".menu").classList.remove("show");
      }
    })
  }
}

class ListChooser {
  chooseCallback;
  loader;
  constructor(loader) {
    this.loader = loader;
    $sel("div.dataList .list").addEventListener("click", (e) => {
      let data = e.path.filter(i => i.dataset && i.dataset["data"])[0];
      this.select(data && data.dataset["data"]);
    })
    $sel("div.dataList .cancelBtn").addEventListener("click", () => {
      this.cancel();
    });
    $sel("div.dataList").addEventListener("click", () => {
      this.cancel();
    });
  }
  async refreshUI() {
    let datas = this.loader && await this.loader() || [];
    let tpl = $sel("#DataListItem").innerHTML;
    $sel("div.dataList > .list").innerHTML = datas.map(d => {
      let data = d.data || d;
      let subtitle = d.subtitle || "";
      return tpl.replace(/{{data}}/g, data)
        .replace("{{subtitle}}", subtitle);
    }).join("");
  }
  select(date) {
    this.chooseCallback && this.chooseCallback(date);
    $sel(".dataList").classList.remove("show")
  }
  cancel() {
    this.chooseCallback && this.chooseCallback();
    $sel(".dataList").classList.remove("show")
  }
  choose() {
    this.refreshUI();
    $sel(".dataList").classList.add("show")

    return new Promise(done => {
      this.chooseCallback = done;
    })
  };
}

class PlaceHolder {
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


class SoundEffect {
  static audio = [];
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
      $sel(".icon-music").style.display="inherit";
    } else {
      $sel(".musicBtn").classList.remove("disabled");
      $sel(".icon-music").style.display="none";
    }
  }
}

window.addEventListener("load", ()=>{
  SoundEffect.bindUI();
  new PlaceHolder();
})